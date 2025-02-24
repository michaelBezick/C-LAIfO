import numpy as np
import torch

from buffers.replay_buffer import AbstractReplayBuffer
from point_cloud_generator import PointCloudGenerator


class EfficientReplayBuffer(AbstractReplayBuffer):
    def __init__(
        self,
        buffer_size,
        batch_size,
        nstep,
        discount,
        frame_stack,
        max_length_point_cloud,
        physics,
        data_specs=None,
    ):
        self.buffer_size = buffer_size
        self.data_dict = {}
        self.index = -1
        self.traj_index = 0
        self.frame_stack = frame_stack
        self._recorded_frames = frame_stack + 1
        self.batch_size = batch_size
        self.nstep = nstep
        self.discount = discount
        self.full = False
        self.point_cloud_generator = PointCloudGenerator(physics)
        # fixed since we can only sample transitions that occur nstep earlier
        # than the end of each episode or the last recorded observation
        self.discount_vec = np.power(discount, np.arange(nstep)).astype("float32")
        self.next_dis = discount**nstep
        self.max_length_point_cloud = max_length_point_cloud

        """
        IMPORTANT CHANGES: POINT CLOUD IS GOING TO BE VARIABLE IN LENGTH, NEED TO HAVE REPLAY BUFFER HANDLE THAT
        Actions are fixed size, so can stay
        """

    def _initial_setup(self, time_step):
        self.index = 0
        self.obs_shape = list(time_step.observation.shape)
        self.ims_channels = self.obs_shape[0] // self.frame_stack #this is correct for normal buffer (not expert)
        self.act_shape = time_step.action.shape

        # self.obs = np.zeros([self.buffer_size, self.ims_channels, *self.obs_shape[1:]], dtype=np.uint8)
        # self.obs = [None] * self.buffer_size
        """THIS IS GOING TO STORE NUMPY ARRAYS FOR POINT CLOUDS"""
        self.obs = np.zeros(
            [self.buffer_size, self.max_length_point_cloud, 3], dtype=np.float32
        )
        self.obs = np.ascontiguousarray(self.obs)

        self.act = np.zeros([self.buffer_size, *self.act_shape], dtype=np.float32)
        self.rew = np.zeros([self.buffer_size], dtype=np.float32)
        self.dis = np.zeros([self.buffer_size], dtype=np.float32)
        # which timesteps can be validly sampled (Not within nstep from end of
        # an episode or last recorded observation)
        self.valid = np.zeros([self.buffer_size], dtype=np.bool_)

    def add_data_point(self, time_step, point_cloud):
        """
        BIG CHANGE, HAVING THE REPLAY BUFFER A NP ARRAY AND WILL 0 PAD / TRUNCATE TO
        MAKE EACH POINT CLOUD A FIXED SIZE
        """
        """
        Expecting each time step to have depth information
        """

        first = time_step.first()

        latest_obs = time_step.observation[-self.ims_channels :] #this is okay ONLY for the normal buffer, doesn't work for expert

        if point_cloud == False:
            # need to convert depth image to point cloud
            if len(np.shape(latest_obs)) == 3:
                if np.shape(latest_obs)[0] == 1:
                    latest_obs = np.squeeze(latest_obs)
                else:
                    print(np.shape(latest_obs))
                    exit()
            latest_obs = np.float32(latest_obs)
            latest_obs = self.point_cloud_generator.depthImageToPointCloud(
                latest_obs, cam_id=0
            )

        num_points = latest_obs.shape[0]

        if num_points > self.max_length_point_cloud:
            latest_obs = latest_obs[: self.max_length_point_cloud]
        elif num_points < self.max_length_point_cloud:
            pad_width = self.max_length_point_cloud - num_points
            latest_obs = np.pad(latest_obs, ((0, pad_width), (0, 0)), mode="constant")

        """LATEST OBS IS ALWAYS POINT CLOUD NO MATTER WHAT"""
        if first:
            # if first observation in a trajectory, record frame_stack copies of it
            end_index = self.index + self.frame_stack
            end_invalid = end_index + self.frame_stack + 1
            if end_invalid > self.buffer_size:
                if end_index > self.buffer_size:
                    end_index = end_index % self.buffer_size
                    self.obs[self.index:self.buffer_size] = latest_obs
                    self.obs[0:end_index] = latest_obs
                    # self.obs[self.index : self.buffer_size] = [
                    #     latest_obs.copy() for _ in range(self.buffer_size - self.index)
                    # ]
                    # self.obs[0:end_index] = [
                    #     latest_obs.copy() for _ in range(end_index)
                    # ]
                    self.full = True
                else:
                    # self.obs[self.index : end_index] = [
                    #     latest_obs.copy() for _ in range(end_index - self.index)
                    # ]
                    self.obs[self.index:end_index] = latest_obs
                end_invalid = end_invalid % self.buffer_size
                self.valid[self.index : self.buffer_size] = False
                self.valid[0:end_invalid] = False
            else:
                # self.obs[self.index : end_index] = [
                #     latest_obs.copy() for _ in range(end_index - self.index)
                # ]
                self.obs[self.index:end_index] = latest_obs
                self.valid[self.index : end_invalid] = False
            self.index = end_index
            self.traj_index = 1
        else:
            np.copyto(self.obs[self.index], latest_obs)
            # self.obs[self.index] = latest_obs.copy()
            np.copyto(self.act[self.index], time_step.action)
            self.rew[self.index] = time_step.reward
            self.dis[self.index] = time_step.discount
            self.valid[(self.index + self.frame_stack) % self.buffer_size] = False
            if self.traj_index >= self.nstep:
                self.valid[(self.index - self.nstep + 1) % self.buffer_size] = True
            self.index += 1
            self.traj_index += 1
            if self.index == self.buffer_size:
                self.index = 0
                self.full = True

    def add(self, time_step, point_cloud=True):
        if self.index == -1:
            self._initial_setup(time_step)
        self.add_data_point(time_step, point_cloud)

    def __next__(
        self,
    ):
        # sample only valid indices
        indices = np.random.choice(self.valid.nonzero()[0], size=self.batch_size)
        return self.gather_nstep_indices(indices)

    def gather_nstep_indices(self, indices):
        n_samples = indices.shape[0]
        all_gather_ranges = np.stack([np.arange(indices[i] - self.frame_stack, indices[i] + self.nstep)
                                  for i in range(n_samples)], axis=0) % self.buffer_size
        gather_ranges = all_gather_ranges[:, self.frame_stack:] # bs x nstep
        obs_gather_ranges = all_gather_ranges[:, :self.frame_stack]
        nobs_gather_ranges = all_gather_ranges[:, -self.frame_stack:]

        all_rewards = self.rew[gather_ranges]

        # Could implement reward computation as a matmul in pytorch for
        # marginal additional speed improvement
        rew = np.sum(all_rewards * self.discount_vec, axis=1, keepdims=True)

        #obs = np.reshape(self.obs[obs_gather_ranges], [n_samples, *self.obs_shape])
        #nobs = np.reshape(self.obs[nobs_gather_ranges], [n_samples, *self.obs_shape])

        #not reshaping because (b,3,d,3) is okay
        obs = self.obs[obs_gather_ranges]
        nobs = self.obs[nobs_gather_ranges]

        act = self.act[indices]
        dis = np.expand_dims(self.next_dis * self.dis[nobs_gather_ranges[:, -1]], axis=-1)

        ret = (obs, act, rew, dis, nobs)

        return ret
        
    def gather_nstep_indices2(self, indices):
        n_samples = indices.shape[0]

        # Compute gather indices
        all_gather_ranges = (
            np.stack(
                [
                    np.arange(indices[i] - self.frame_stack, indices[i] + self.nstep)
                    for i in range(n_samples)
                ],
                axis=0,
            )
            % self.buffer_size
        )

        gather_ranges = all_gather_ranges[:, self.frame_stack :]  # bs x nstep
        obs_gather_ranges = all_gather_ranges[:, : self.frame_stack]
        nobs_gather_ranges = all_gather_ranges[:, -self.frame_stack :]

        # Collect rewards
        all_rewards = self.rew[gather_ranges]
        rew = np.sum(all_rewards * self.discount_vec, axis=1, keepdims=True)

        # Gather observations (variable-sized point clouds)
        obs = [[self.obs[i] for i in idx_row] for idx_row in obs_gather_ranges]
        nobs = [[self.obs[i] for i in idx_row] for idx_row in nobs_gather_ranges]

        # Determine max number of points across batch
        max_points = max(
            max(frame.shape[0] for sample in obs for frame in sample),
            max(frame.shape[0] for sample in nobs for frame in sample),
        )

        # Pad and convert to NumPy arrays
        def pad_point_clouds(data):
            padded_data = []
            for sample in data:
                padded_sample = []
                for frame in sample:
                    N, D = (
                        frame.shape
                    )  # Get the number of points and dimensions (should be 3)
                    pad_width = max_points - N
                    if pad_width > 0:
                        padded_frame = np.pad(
                            frame, ((0, pad_width), (0, 0)), mode="constant"
                        )
                    else:
                        padded_frame = (
                            frame  # No padding needed if it's already max_points
                        )
                    padded_sample.append(padded_frame)
                padded_data.append(padded_sample)
            return np.array(padded_data)

        obs_padded = pad_point_clouds(
            obs
        )  # Shape: (batch_size, frame_stack, max_points, 3)
        nobs_padded = pad_point_clouds(
            nobs
        )  # Shape: (batch_size, frame_stack, max_points, 3)

        act = self.act[indices]
        dis = np.expand_dims(
            self.next_dis * self.dis[nobs_gather_ranges[:, -1]], axis=-1
        )

        return obs_padded, act, rew, dis, nobs_padded

    """
    def gather_nstep_indices(self, indices):
        n_samples = indices.shape[0]
        all_gather_ranges = np.stack([np.arange(indices[i] - self.frame_stack, indices[i] + self.nstep)
                                  for i in range(n_samples)], axis=0) % self.buffer_size
        gather_ranges = all_gather_ranges[:, self.frame_stack:] # bs x nstep
        obs_gather_ranges = all_gather_ranges[:, :self.frame_stack]
        nobs_gather_ranges = all_gather_ranges[:, -self.frame_stack:]

        all_rewards = self.rew[gather_ranges]

        # Could implement reward computation as a matmul in pytorch for
        # marginal additional speed improvement
        rew = np.sum(all_rewards * self.discount_vec, axis=1, keepdims=True)

        #obs = np.reshape(self.obs[obs_gather_ranges], [n_samples, *self.obs_shape])
        #nobs = np.reshape(self.obs[nobs_gather_ranges], [n_samples, *self.obs_shape])
        obs = [[self.obs[i] for i in idx_row] for idx_row in obs_gather_ranges]
        nobs = [[self.obs[i] for i in idx_row] for idx_row in nobs_gather_ranges]

        #obs = np.array([[self.obs[i] for i in idx_row] for idx_row in obs_gather_ranges], dtype=object)
        #nobs = np.array([[self.obs[i] for i in idx_row] for idx_row in nobs_gather_ranges], dtype=object)

        # Determine the max number of points across the batch
        max_points = max(max(frame.shape[0] for sample in obs for frame in sample),
                         max(frame.shape[0] for sample in nobs for frame in sample))

        def pad_point_clouds(data):

            return np.array([
                np.array([
                    np.vstack([frame, np.zeros((max_points - frame.shape[0], 3))]) for frame in sample
                ]) for sample in data
            ])

        obs_padded = pad_point_clouds(obs)  # Shape: (batch_size, frame_stack, max_points, 3)
        nobs_padded = pad_point_clouds(nobs)  # Shape: (batch_size, frame_stack, max_points, 3)

        act = self.act[indices]
        dis = np.expand_dims(self.next_dis * self.dis[nobs_gather_ranges[:, -1]], axis=-1)

        return (obs_padded, act, rew, dis, nobs_padded)

        # Convert to tensor with padding
        obs_tensor = torch.stack([
            torch.stack([torch.cat([torch.tensor(frame), torch.zeros((max_points - frame.shape[0], 3))]) for frame in sample])
            for sample in obs
        ])

        obs_tensor = obs_tensor.detach().numpy()

        nobs_tensor = torch.stack([
            torch.stack([torch.cat([torch.tensor(frame), torch.zeros((max_points - frame.shape[0], 3))]) for frame in sample])
            for sample in nobs
        ])

        nobs_tensor = nobs_tensor.detach().numpy()

        #obs = np.reshape(obs, [n_samples, *self.obs_shape])
        #nobs = np.reshape(nobs, [n_samples, *self.obs_shape])


        act = self.act[indices]
        dis = np.expand_dims(self.next_dis * self.dis[nobs_gather_ranges[:, -1]], axis=-1)

        ret = (obs_tensor, act, rew, dis, nobs_tensor)
        return ret
        """

    def gather_images(self):
        indices = np.random.choice(self.valid.nonzero()[0], size=self.batch_size)
        return self.obs[indices, :, :, :]

    def __len__(self):
        if self.full:
            return self.buffer_size
        else:
            return self.index

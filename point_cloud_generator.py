#!/usr/bin/env python3
import math
import pdb
import time

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from dm_control import suite
from dm_control._render.executor import render_executor
from dm_control.suite.walker import Physics
from PIL import Image as PIL_Image
from BoundingBoxSkeleton.functions import bounding_box_centers
from sklearn.decomposition import PCA

from L1MedialSkeleton.functions import l1_medial_skeleton

"""
Generates numpy rotation matrix from quaternion

@param quat: w-x-y-z quaternion rotation tuple

@return np_rot_mat: 3x3 rotation matrix as numpy array
"""

def transform_to_pca_basis(points, pca):
    return pca.transform(points)  # Project points onto PCA basis

def compute_pca(points):
    pca = PCA(n_components=3)
    pca.fit(points)
    components = pca.components_  # Principal components (3x3 matrix)
    explained_variance = pca.explained_variance_ratio_  # Variance explained by each PC
    return pca, components, explained_variance


def quat2Mat(quat):
    if len(quat) != 4:
        print("Quaternion", quat, "invalid when generating transformation matrix.")
        raise ValueError

    # Note that the following code snippet can be used to generate the 3x3
    #    rotation matrix, we don't use it because this file should not depend
    #    on mujoco.
    """
    from mujoco_py import functions
    res = np.zeros(9)
    functions.mju_quat2Mat(res, camera_quat)
    res = res.reshape(3,3)
    """

    # This function is lifted directly from scipy source code
    # https://github.com/scipy/scipy/blob/v1.3.0/scipy/spatial/transform/rotation.py#L956
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    rot_mat_arr = [
        x2 - y2 - z2 + w2,
        2 * (xy - zw),
        2 * (xz + yw),
        2 * (xy + zw),
        -x2 + y2 - z2 + w2,
        2 * (yz - xw),
        2 * (xz - yw),
        2 * (yz + xw),
        -x2 - y2 + z2 + w2,
    ]
    np_rot_mat = rotMatList2NPRotMat(rot_mat_arr)
    return np_rot_mat


"""
Generates numpy rotation matrix from rotation matrix as list len(9)

@param rot_mat_arr: rotation matrix in list len(9) (row 0, row 1, row 2)

@return np_rot_mat: 3x3 rotation matrix as numpy array
"""


def rotMatList2NPRotMat(rot_mat_arr):
    np_rot_arr = np.array(rot_mat_arr)
    np_rot_mat = np_rot_arr.reshape((3, 3))
    return np_rot_mat


"""
Generates numpy transformation matrix from position list len(3) and 
    numpy rotation matrix

@param pos:     list len(3) containing position
@param rot_mat: 3x3 rotation matrix as numpy array

@return t_mat:  4x4 transformation matrix as numpy array
"""


def posRotMat2Mat(pos, rot_mat):
    t_mat = np.eye(4)
    t_mat[:3, :3] = rot_mat
    t_mat[:3, 3] = np.array(pos)
    return t_mat


"""
Generates Open3D camera intrinsic matrix object from numpy camera intrinsic
    matrix and image width and height

@param cam_mat: 3x3 numpy array representing camera intrinsic matrix
@param width:   image width in pixels
@param height:  image height in pixels

@return t_mat:  4x4 transformation matrix as numpy array
"""


def cammat2o3d(cam_mat, width, height):
    cx = cam_mat[0, 2]
    fx = cam_mat[0, 0]
    cy = cam_mat[1, 2]
    fy = cam_mat[1, 1]

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


#
# and combines them into point clouds
"""
Class that renders depth images in MuJoCo, processes depth images from
    multiple cameras, converts them to point clouds, and processes the point
    clouds
"""


class PointCloudGenerator(object):
    """
    initialization function

    @param sim:       MuJoCo simulation object
    @param min_bound: If not None, list len(3) containing smallest x, y, and z
        values that will not be cropped
    @param max_bound: If not None, list len(3) containing largest x, y, and z
        values that will not be cropped
    """

    def __init__(self, sim: Physics, min_bound=None, max_bound=None):
        super(PointCloudGenerator, self).__init__()

        self.sim = sim

        self.img_width = 64
        self.img_height = 64

        self.cam_names = [i for i in range(len(self.sim.model.cam_bodyid))]

        self.target_bounds = None
        if min_bound != None and max_bound != None:
            self.target_bounds = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=min_bound, max_bound=max_bound
            )

        # List of camera intrinsic matrices
        self.cam_mats = []
        for cam_id in range(len(self.cam_names)):
            fovy = math.radians(self.sim.model.cam_fovy[cam_id])
            f = self.img_height / (2 * math.tan(fovy / 2))
            cam_mat = np.array(
                ((f, 0, self.img_width / 2), (0, f, self.img_height / 2), (0, 0, 1))
            )
            self.cam_mats.append(cam_mat)

    def depthImageToPointCloud(
        self,
        depth_img,
        cam_id,
        max_depth=6,
        down_sample_voxel_size=-1,
        skeleton=False,
        PCA=True,
    ) -> np.ndarray:
        """
        @param down_sample_voxel_size: put to -1 to disable downsampling

        """


        od_cammat = cammat2o3d(self.cam_mats[cam_id], self.img_width, self.img_height)

        depth_img[depth_img >= max_depth] = 0

        od_depth = o3d.geometry.Image(np.ascontiguousarray(depth_img))

        o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(od_depth, od_cammat)

        cam_pos = self.sim.model.cam_pos[cam_id]
        c2b_r = rotMatList2NPRotMat(self.sim.model.cam_mat0[cam_id])
        b2w_r = quat2Mat([0, 1, 0, 0])
        c2w_r = np.matmul(c2b_r, b2w_r)
        c2w = posRotMat2Mat(cam_pos, c2w_r)
        transformed_cloud = o3d_cloud.transform(c2w)

        if self.target_bounds != None:
            transformed_cloud = transformed_cloud.crop(self.target_bounds)

        points = np.asarray(transformed_cloud.points)

        # not centroid shifting and unit sphere scaling anymore

        """
        #centroid shift
        center = np.mean(points, axis=0)
        centered_points = points - center

        #unit sphere magnitude
        magnitudes = np.linalg.norm(centered_points, axis=1)
        max_magnitude = np.max(magnitudes)
        
        if max_magnitude > 0:
            normalized_points = centered_points / max_magnitude
            transformed_cloud.points = o3d.utility.Vector3dVector(normalized_points)
        else:
            transformed_cloud.points = o3d.utility.Vector3dVector(centered_points)
        """

        bounding_box = False


        if bounding_box:

            centers = bounding_box_centers(transformed_cloud)
            np.random.shuffle(centers)

            if np.isnan(centers).any() or np.isinf(centers).any():

                print(centers)
                raise ValueError("bounding_box_centers produced NaN or Inf values!")

            if np.abs(centers).max() > 100:  # Adjust threshold based on environment
                raise ValueError(f"bounding_box_centers output has extreme values: {centers}")

            return centers.astype(np.float32)


        if skeleton:

            transformed_cloud.points = o3d.utility.Vector3dVector(
                points[points[:, 2] >= -1.2]  # filtering points
            )

            transformed_cloud = transformed_cloud.voxel_down_sample(
                voxel_size=0.17
            )

            points = np.asarray(transformed_cloud.points)

            #transformed_cloud pc_skeleton = l1_medial_skeleton(transformed_cloud)
            np.random.shuffle(points)  # so truncation isn't biased

            return points.astype(np.float32)

        transformed_cloud.points = o3d.utility.Vector3dVector(points)

        if down_sample_voxel_size != -1:

            transformed_cloud = transformed_cloud.voxel_down_sample(
                voxel_size=down_sample_voxel_size
            )

        points = np.asarray(transformed_cloud.points)

        if PCA:
            pca, components, explained_variance = compute_pca(points)
            points = transform_to_pca_basis(points, pca)
            """NEED TO HAVE DATA STRUCTURE ALLOW FOR 4 CANONICAL POSES"""
            """ALSO NEED TO CALCULATE ALL 4"""

        np.random.shuffle(points)  # so truncation isn't biased

        return points.astype(np.float32)

    def generateCroppedPointCloud(self, save_img_dir=None, fast=True):
        o3d_clouds = []
        cam_poses = []
        for cam_i in range(len(self.cam_names)):
            # Render and optionally save image from camera corresponding to cam_i
            depth_img = self.captureImage(cam_i)
            # If directory was provided, save color and depth images
            #    (overwriting previous)
            if fast == False:
                if save_img_dir != None:
                    self.saveImg(depth_img, save_img_dir, "depth_test_" + str(cam_i))
                    color_img = self.captureImage(cam_i, False)
                    self.saveImg(color_img, save_img_dir, "color_test_" + str(cam_i))

            # convert camera matrix and depth image to Open3D format, then
            #    generate point cloud
            od_cammat = cammat2o3d(
                self.cam_mats[cam_i], self.img_width, self.img_height
            )

            max_depth = 6

            depth_img[depth_img >= max_depth] = 0

            od_depth = o3d.geometry.Image(np.ascontiguousarray(depth_img))

            o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(
                od_depth, od_cammat
            )

            # print("Original point count:", np.asarray(o3d_cloud.points).shape[0])
            #
            # o3d_cloud = o3d_cloud.voxel_down_sample(voxel_size=0.05)
            #
            # print("Downsampled point count:", np.asarray(o3d_cloud.points).shape[0])

            cam_pos = self.sim.model.cam_pos[cam_i]

            c2b_r = rotMatList2NPRotMat(self.sim.model.cam_mat0[cam_i])

            # In MuJoCo, we assume that a camera is specified in XML as a body
            #    with pose p, and that that body has a camera sub-element
            #    with pos and euler 0.
            #    Therefore, camera frame with body euler 0 must be rotated about
            #    x-axis by 180 degrees to align it with the world frame.
            """"""
            b2w_r = quat2Mat([0, 1, 0, 0])
            """"""
            c2w_r = np.matmul(c2b_r, b2w_r)
            c2w = posRotMat2Mat(cam_pos, c2w_r)
            transformed_cloud = o3d_cloud.transform(c2w)

            # If both minimum and maximum bounds are provided, crop cloud to fit
            #    inside them.
            if self.target_bounds != None:
                transformed_cloud = transformed_cloud.crop(self.target_bounds)

            # Estimate normals of cropped cloud, then flip them based on camera
            #    position.
            """
            transformed_cloud.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.03, max_nn=250
                )
            )
            transformed_cloud.orient_normals_towards_camera_location(cam_pos)
            """

            o3d_clouds.append(transformed_cloud)

        combined_cloud = o3d.geometry.PointCloud()
        for i, cloud in enumerate(o3d_clouds):
            combined_cloud += cloud

        # scale the point cloud to have max magnitude of 1 and shift to have mean 0,0,0
        points = np.asarray(combined_cloud.points)

        center = np.mean(points, axis=0)
        centered_points = points - center
        print("centroid: ", np.mean(centered_points, axis=0))

        magnitudes = np.linalg.norm(centered_points, axis=1)
        max_magnitude = np.max(magnitudes)
        print("max_magnitude: ", max_magnitude)

        if max_magnitude > 0:
            normalized_points = centered_points / max_magnitude
            combined_cloud.points = o3d.utility.Vector3dVector(normalized_points)
        else:
            combined_cloud.points = o3d.utility.Vector3dVector(centered_points)

        return combined_cloud

    # https://github.com/htung0101/table_dome/blob/master/table_dome_calib/utils.py#L160
    def depthimg2Meters(self, depth):
        extent = self.sim.model.stat.extent
        near = self.sim.model.vis.map.znear * extent
        far = self.sim.model.vis.map.zfar * extent
        image = near / (1 - depth * (1 - near / far))
        return image

    def verticalFlip(self, img):
        return img
        # return np.flip(img, axis=0)

    # Render and process an image
    def captureImage(self, cam_ind, capture_depth=True):

        img = self.sim.render(
            width=self.img_width,
            height=self.img_height,
            camera_id=self.cam_names[cam_ind],
            depth=False,
        )

        depth = self.sim.render(
            width=self.img_width,
            height=self.img_height,
            camera_id=self.cam_names[cam_ind],
            depth=True,
        )

        if capture_depth:
            depth = self.verticalFlip(depth)
            # real_depth = self.depthimg2Meters(depth)
            real_depth = depth

            return real_depth
        else:
            # Rendered images appear to be flipped about vertical axis
            return self.verticalFlip(img)

    # Normalizes an image so the maximum pixel value is 255,
    # then writes to file
    def saveImg(self, img, filepath, filename):
        normalized_image = img / img.max() * 255
        normalized_image = normalized_image.astype(np.uint8)
        im = PIL_Image.fromarray(normalized_image)
        im.save(filepath + "/" + filename + ".jpg")

    def save_point_cloud_as_image(
        self, point_cloud, output_image="point_cloud_image.png"
    ):
        """Saves a 2D projection of the point cloud to an image using offscreen rendering."""
        # Offscreen rendering
        renderer = o3d.visualization.rendering.OffscreenRenderer(640, 480)

        # Add the point cloud to the scene
        renderer.scene.add_geometry(
            "point_cloud", point_cloud, o3d.visualization.rendering.MaterialRecord()
        )

        # Set the camera perspective
        renderer.scene.camera.look_at([0, 0, 0], [0, 0, 1], [0, 1, 0])

        # Render the scene and save as image
        image = renderer.render_to_image()
        o3d.io.write_image(output_image, image)

        # Clean up
        print(f"Point cloud projection saved to {output_image}")

    def save_point_cloud(
        self, point_cloud, is_point_cloud=True, output_file="point_cloud.ply"
    ):

        if not is_point_cloud:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)

            point_cloud = pcd

        # Save the point cloud
        o3d.io.write_point_cloud("./point_cloud_images/" + output_file, point_cloud)
        print(f"Point cloud saved to {output_file}")


if __name__ == "__main__":
    env = suite.load(domain_name="walker", task_name="walk")
    physics = env.physics
    point_cloud_generator = PointCloudGenerator(physics)

    depth1 = physics.render(
        width=64,
        height=64,
        camera_id=0,
        depth=True,
    )

    depth2 = physics.render(
        width=64,
        height=64,
        camera_id=1,
        depth=True,
    )

    combined_cloud = o3d.geometry.PointCloud()

    point_cloud1 = point_cloud_generator.depthImageToPointCloud(
        depth1, cam_id=0, down_sample_voxel_size=0.03
    )

    # num_points = point_cloud1.shape[0]
    # truncate = num_points // 2
    # point_cloud1 = point_cloud1[:truncate, :]

    print("point cloud 1 points: ", point_cloud1.shape)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(point_cloud1)
    point_cloud_generator.save_point_cloud(
        pcd1, is_point_cloud=True, output_file="view1.ply"
    )

    point_cloud2 = point_cloud_generator.depthImageToPointCloud(
        depth2, cam_id=1, down_sample_voxel_size=0.03
    )

    # num_points = point_cloud2.shape[0]
    # truncate = num_points // 2
    #
    # point_cloud2 = point_cloud2[:truncate, :]

    print("point cloud 2 points: ", point_cloud2.shape)
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(point_cloud2)
    point_cloud_generator.save_point_cloud(
        pcd2, is_point_cloud=True, output_file="view2.ply"
    )

    exit()

    combined_cloud += pcd1
    combined_cloud += pcd2

    point_cloud_generator.save_point_cloud(
        combined_cloud, is_point_cloud=True, output_file="combined.ply"
    )

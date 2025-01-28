import math
import matplotlib.pyplot as plt
import numpy as np
from dm_control import suite
import open3d as o3d

from dm_control._render.executor import render_executor

def quat2Mat(quat):
    """Converts quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    w, x, y, z = quat
    x2, y2, z2 = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    return np.array([
        [1 - 2 * (y2 + z2), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (x2 + z2), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (x2 + y2)],
    ])

def depthimg2Meters(depth, near, far):
    """Converts normalized depth image to real-world distances."""
    return near / (1 - depth * (1 - near / far))

class DMControlPointCloudGenerator:
    def __init__(self, physics, img_width=640, img_height=480, min_bound=None, max_bound=None):
        self.physics = physics
        self.img_width = img_width
        self.img_height = img_height
        self.min_bound = min_bound
        self.max_bound = max_bound

        self.target_bounds = None
        if min_bound is not None and max_bound is not None:
            self.target_bounds = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)

    def generate_point_cloud(self, depth, camera_id=0):
        print("depth", depth)

        # Convert depth to real-world distances
        near = self.physics.model.vis.map.znear
        far = self.physics.model.vis.map.zfar

        print("near", near)
        print("far", far)


        #depth_in_meters = depthimg2Meters(depth, near, far)
        depth_in_meters = depth
        print("depth in meters", depth_in_meters)

        # Get camera intrinsics
        fovy = math.radians(self.physics.model.cam_fovy[camera_id])
        f = self.img_height / (2 * math.tan(fovy / 2))
        cam_mat = np.array([
            [f, 0, self.img_width / 2],
            [0, f, self.img_height / 2],
            [0, 0, 1]
        ])
        print("cam mat", cam_mat)
        o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(self.img_width, self.img_height, f, f, self.img_width / 2, self.img_height / 2)

        # Create point cloud
        #depth_img = o3d.geometry.Image(depth_in_meters)
        depth_img = o3d.geometry.Image(np.ascontiguousarray(depth_in_meters))
        
        point_cloud = o3d.geometry.PointCloud.create_from_depth_image(depth_img, o3d_intrinsics)

        # Get camera extrinsics
        cam_pos = self.physics.model.cam_pos[camera_id]
        cam_quat = self.physics.model.cam_quat[camera_id]
        cam_rot = quat2Mat(cam_quat)
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = cam_rot
        extrinsics[:3, 3] = cam_pos

        # Transform point cloud to world coordinates
        point_cloud.transform(extrinsics)

        #ignore point cloud points that are far away
        points = np.asarray(point_cloud.points)
        valid_indices = points[:, 2] < 6
        filtered_points = points[valid_indices]
        point_cloud.points = o3d.utility.Vector3dVector(filtered_points)

        # Crop point cloud if bounds are specified
        if self.target_bounds is not None:
            point_cloud = point_cloud.crop(self.target_bounds)

        return point_cloud


    def generate_point_cloud_old(self, camera_id=0):
        """Generates a 3D point cloud from a specified camera."""
        # Render depth image
        depth = self.physics.render(width=self.img_width, height=self.img_height, camera_id=camera_id, depth=True)
        print("depth", depth)

        # Convert depth to real-world distances
        near = self.physics.model.vis.map.znear
        far = self.physics.model.vis.map.zfar

        print("near", near)
        print("far", far)


        #depth_in_meters = depthimg2Meters(depth, near, far)
        depth_in_meters = depth
        print("depth in meters", depth_in_meters)

        # Get camera intrinsics
        fovy = math.radians(self.physics.model.cam_fovy[camera_id])
        f = self.img_height / (2 * math.tan(fovy / 2))
        cam_mat = np.array([
            [f, 0, self.img_width / 2],
            [0, f, self.img_height / 2],
            [0, 0, 1]
        ])
        print("cam mat", cam_mat)
        o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(self.img_width, self.img_height, f, f, self.img_width / 2, self.img_height / 2)

        # Create point cloud
        #depth_img = o3d.geometry.Image(depth_in_meters)
        depth_img = o3d.geometry.Image(np.ascontiguousarray(depth_in_meters))
        
        point_cloud = o3d.geometry.PointCloud.create_from_depth_image(depth_img, o3d_intrinsics)

        # Get camera extrinsics
        cam_pos = self.physics.model.cam_pos[camera_id]
        cam_quat = self.physics.model.cam_quat[camera_id]
        cam_rot = quat2Mat(cam_quat)
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = cam_rot
        extrinsics[:3, 3] = cam_pos

        # Transform point cloud to world coordinates
        point_cloud.transform(extrinsics)

        # Crop point cloud if bounds are specified
        if self.target_bounds is not None:
            point_cloud = point_cloud.crop(self.target_bounds)

        return point_cloud

    def save_point_cloud(self, point_cloud, output_file="point_cloud.ply"):

        # Save the point cloud
        o3d.io.write_point_cloud("./point_cloud_images/" + output_file, point_cloud)
        print(f"Point cloud saved to {output_file}")

    def save_point_cloud_as_image(self, point_cloud, output_image="point_cloud_image.png"):
        """Saves a 2D projection of the point cloud to an image using offscreen rendering."""
        # Offscreen rendering
        renderer = o3d.visualization.rendering.OffscreenRenderer(640, 480)

        # Add the point cloud to the scene
        renderer.scene.add_geometry("point_cloud", point_cloud, o3d.visualization.rendering.MaterialRecord())

        # Set the camera perspective
        renderer.scene.camera.look_at([0, 0, 0], [0, 0, 1], [0, 1, 0])

        # Render the scene and save as image
        image = renderer.render_to_image()
        o3d.io.write_image(output_image, image)


        # Clean up
        renderer.close()
        print(f"Point cloud projection saved to {output_image}")

# Example usage
if __name__ == "__main__":
    env = suite.load(domain_name="walker", task_name="walk")
    physics = env.physics
    img_width=640
    img_height=480
    min_bound=None
    max_bound=None
    camera_id=0

    depth = physics.render(width=img_width, height=img_height, camera_id=camera_id, depth=True)
    rgb_image = physics.render(height=480, width=640, camera_id=0)

    min_depth, max_depth = 1, 6
    #filtering all values greater than 6 to be 0
    depth[depth >= max_depth] = 0
    # clipped_depth = np.clip(depth, min_depth, max_depth)
    normalized_depth = (depth - min_depth) / (np.max(depth) - min_depth)
    depth_map = normalized_depth


    pc_generator = DMControlPointCloudGenerator(physics)

    # Generate the point cloud
    point_cloud = pc_generator.generate_point_cloud(depth_map)
    print("pc points", len(point_cloud.points))
    # Save the point cloud as a 2D projection image
    pc_generator.save_point_cloud(point_cloud)
# Display the RGB image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_image)
    plt.title("RGB Image")

# Display the depth map
    plt.subplot(1, 2, 2)
    plt.imshow(depth_map, cmap="viridis")
    plt.colorbar(label="Depth")
    plt.title("Depth Map")

    plt.savefig("./depth_test/3rd.png", dpi=300)

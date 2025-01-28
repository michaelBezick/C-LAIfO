import open3d as o3d

# Load the .ply file
point_cloud = o3d.io.read_point_cloud("./point_cloud.ply")

# Visualize the point cloud
o3d.visualization.draw_plotly([point_cloud], window_name="Point Cloud")

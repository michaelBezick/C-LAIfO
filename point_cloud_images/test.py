import open3d as o3d

# Load the open point cloud
pcd = o3d.io.read_point_cloud("view1.ply")
# o3d.visualization.draw_plotly([pcd], window_name="Point Cloud")

# Ensure point cloud is not empty
if len(pcd.points) == 0:
    raise ValueError("Error: The point cloud is empty. Please check your file.")

# Estimate normals (needed for Poisson)
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=50)  # Increase K for better estimation
)

# Orient normals consistently
pcd.orient_normals_consistent_tangent_plane(k=100)

# Debug: Check if normals are assigned
if len(pcd.normals) == 0:
    raise ValueError("Error: Normal estimation failed. Check input data.")

# Poisson Surface Reconstruction
mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

# Visualize
o3d.visualization.draw_plotly([mesh], window_name="Point Cloud")
# o3d.visualization.draw_geometries([mesh], window_name="Poisson Mesh")

# Save closed mesh
o3d.io.write_triangle_mesh("closed_mesh.ply", mesh)


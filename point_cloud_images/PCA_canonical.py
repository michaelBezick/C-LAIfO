import open3d as o3d
import numpy as np
import plotly.graph_objects as go

point_cloud = o3d.io.read_point_cloud("./view1.ply")
points = np.asarray(point_cloud.points)

o3d.io.write_point_cloud("filtered_point_cloud.ply", pcd_filtered)

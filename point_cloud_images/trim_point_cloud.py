import open3d as o3d
import numpy as np
import plotly.graph_objects as go


point_cloud = o3d.io.read_point_cloud("./view1.ply")
points = np.asarray(point_cloud.points)
x_min, y_min, z_min = points.min(axis=0)
x_max, y_max, z_max = points.max(axis=0)


valid_indices = np.where(points[:, 2] >= -1.2)[0]
pcd_filtered = point_cloud.select_by_index(valid_indices)

filtered_points = np.asarray(pcd_filtered.points)
x, y, z = filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2]

# Create a Plotly scatter plot
fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers',
                                   marker=dict(size=2, color=z, colorscale='Viridis'))])

# Set axis ranges and keep the original scale
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max]),
        zaxis=dict(range=[z_min, z_max]),
        aspectmode="manual",  # Prevents auto-scaling
        aspectratio=dict(x=(x_max - x_min), y=(y_max - y_min), z=(z_max - z_min))
    ),
    title="Filtered Point Cloud (Fixed Scale)"
)

# Save filtered point cloud
o3d.io.write_point_cloud("filtered_point_cloud.ply", pcd_filtered)

# Show Plotly visualization
fig.show()

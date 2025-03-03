import open3d as o3d


import numpy as np
from pc_skeletor import LBC
import time

point_cloud = o3d.io.read_point_cloud("./point_cloud_images/random_point_cloud_filtered.ply")
print(np.asarray(point_cloud.points).shape)
exit()

time1 = time.time()
lbc = LBC(point_cloud=point_cloud, down_sample=0.001)
lbc.extract_skeleton()
lbc.extract_topology()
time2 = time.time()

print(time2 - time1)
# lbc.visualize()
lbc.export_results('./output')
# lbc.animate(init_rot=np.asarray([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
#             steps=300,
#             output='./output')

import time

import numpy as np
import open3d as o3d
from pc_skeletor import LBC

point_cloud = o3d.io.read_point_cloud(
    "./point_cloud_images/random_point_cloud_filtered.ply"
)

time1 = time.time()


lbc = LBC(
    point_cloud=point_cloud,
    init_contraction=0.2,  # Less aggressive contraction
    init_attraction=1.2,  # Higher attraction to preserve small branches
    max_contraction=256,  # Prevent collapse into spine
    max_attraction=10000,  # Stronger attraction to keep legs
    step_wise_contraction_amplification=2.0,  # Allow local refinements
    termination_ratio=0.01,  # Allow longer contraction process
    max_iteration_steps=300,  # More steps for refining
    down_sample=0.00005,  # Capture fine details
    filter_nb_neighbors=5,  # Less pruning of small structures
    filter_std_ratio=5.0,
    debug=False,
    verbose=False,
)

lbc.extract_skeleton()
lbc.extract_topology()
time2 = time.time()

print("TIMETIMETIMETIME: ", time2 - time1)
lbc.visualize()
lbc.export_results("./output")
lbc.animate(
    init_rot=np.asarray([[1, 0, 0], [0, 0, 1], [0, 1, 0]]), steps=300, output="./output"
)

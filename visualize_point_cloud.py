import open3d as o3d
from pathlib import Path
from pointcloud_to_surface_models import process_model

# Load the PLY file
#process_model(Path("models/dragon/dragon_vrip.ply"), "dragon95", 95, 2)

pcd = o3d.io.read_point_cloud("output/dragon99.9/dense_output.ply")
#pcd = o3d.io.read_point_cloud("models/processed_ModelNet40/dragon_vrip.ply")

# Visualize
o3d.visualization.draw_geometries([pcd])

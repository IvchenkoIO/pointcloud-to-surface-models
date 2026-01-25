from open3d import io, geometry, visualization
import numpy as np
import threading
from pathlib import Path
import os
import upsampling as up
import upsampling_helper_funcs as up_h
from comparison import evaluate



armadillo = Path("models/armadillo/Armadillo.ply") # remove 90+%

buddha_full = Path("models/buddha/happy_vrip.ply") # remove 95+%
buddha_small = Path("models/buddha/happy_vrip_res2.ply")
buddha_smaller = Path("models/buddha/happy_vrip_res3.ply")
buddha_smallest = Path("models/buddha/happy_vrip_res4.ply") # maybe remove like 50% to add some randomness

bunny_full = Path("models/bunny/bun_zipper.ply") # remove 80+%
bunny_small = Path("models/bunny/bun_zipper_res2.ply")
bunny_smaller = Path("models/bunny/bun_zipper_res3.ply")
bunny_smallest = Path("models/bunny/bun_zipper_res4.ply") # good as is

dragon_full = Path("models/dragon/dragon_vrip.ply") # remove 95+%
dragon_small = Path("models/dragon/dragon_vrip_res2.ply")
dragon_smaller = Path("models/dragon/dragon_vrip_res3.ply")
dragon_smallest = Path("models/dragon/dragon_vrip_res4.ply") # maybe remove like 50% to add some randomness

lucy = "models/lucy/lucy.ply" # It's extremely large, remove like 99.9% points

output_path = Path("output/")
extension = ".ply"

def load_and_reduce_point_cloud(cloud, percentage):
    num_points = len(cloud.points)

    keep_ratio = 1 - (percentage / 100)
    num_keep = int(num_points * keep_ratio)

    indices = np.random.choice(num_points, num_keep, replace=False)
    reduced_cloud = cloud.select_by_index(indices)
    return reduced_cloud


def visualize_point_cloud(cloud):
    visualization.draw_geometries([cloud])


def stats(name, pcd):
    aabb = pcd.get_axis_aligned_bounding_box()
    mins = aabb.get_min_bound()
    maxs = aabb.get_max_bound()
    center = aabb.get_center()
    extent = aabb.get_extent()
    diag = np.linalg.norm(extent)
    print(f"\n[{name}]")
    print(f"  min:    {mins}")
    print(f"  max:    {maxs}")
    print(f"  center: {center}")
    print(f"  extent: {extent}  (diag ≈ {diag:.6f})")
    return aabb


def process_model(filename, model_name, percentage_to_remove, upscale):
    full_cloud = io.read_point_cloud(filename)
    sparse = load_and_reduce_point_cloud(full_cloud, percentage_to_remove)

    os.makedirs(f"output/{model_name}", exist_ok=True)
    io.write_point_cloud(f"output/{model_name}/sparse_output.ply", sparse)

    # Fill till same size
    #points_to_add = len(full_cloud.points) - len(sparse.points)
    # Upscale by (X + 1)
    points_to_add = int(len(sparse.points) * (upscale - 1))
    # Add X points
    #points_to_add = 10000

    #print(f"Sparse: {len(reduced_cloud.points)} points")
    #up.upsampling_alg(reduced_cloud,str(output_path),str(filename.stem),extension)
    #visualize_point_cloud(reduced_cloud)
    up.upsampling_self(model_name, filename=str(f"output/{model_name}/sparse_output.ply"), num_iterations=points_to_add)
    dense = io.read_point_cloud(f"output/{model_name}/dense_output.ply")
    
    # Compare result to ground truth
    evaluate(full_cloud, dense, model_name)

    '''
    # color for contrast
    sparse.paint_uniform_color([1.0, 0.3, 0.1])  # red
    dense.paint_uniform_color([0.1, 0.6, 1.0])  # blue

    aabb_s = stats("sparse", sparse)
    aabb_d = stats("dense", dense)

    # Spacing
    extent_s = aabb_s.get_extent()
    extent_d = aabb_d.get_extent()
    shift = max(extent_s[0], extent_d[0]) * 1.2  # shift along X

    # Translate dense cloud to the right
    dense.translate([shift, 0, 0])

    # Combined AABB for view centering
    mins = np.minimum(aabb_s.get_min_bound(), aabb_d.get_min_bound() + [shift, 0, 0])
    maxs = np.maximum(aabb_s.get_max_bound(), aabb_d.get_max_bound() + [shift, 0, 0])
    combo = geometry.AxisAlignedBoundingBox(mins, maxs)

    # --- Visualization ---
    vis = visualization.Visualizer()
    vis.create_window(window_name="Sparse (Red, Left) vs Dense (Blue, Right)")
    vis.add_geometry(sparse)
    vis.add_geometry(dense)

    opt = vis.get_render_option()
    opt.point_size = 3.0
    opt.background_color = np.array([1, 1, 1])

    vc = vis.get_view_control()
    vc.set_lookat(combo.get_center())
    vc.set_front([0.0, 0.0, -1.0])
    vc.set_up([0.0, -1.0, 0.0])
    vc.set_zoom(0.7)

    axes = geometry.TriangleMesh.create_coordinate_frame(
        size=max(combo.get_extent()) * 0.1, origin=combo.get_center()
    )
    vis.add_geometry(axes)

    vis.run()
    vis.destroy_window()
    '''


def diff_num_points():
    process_model(Path("models/processed_ModelNet40/Armadillo.ply"), "Armadillo95", 95, 2)
    process_model(Path("models/processed_ModelNet40/Armadillo.ply"), "Armadillo99", 99, 10)
    process_model(Path("models/processed_ModelNet40/Armadillo.ply"), "Armadillo99.5", 99.5, 20)
    process_model(Path("models/processed_ModelNet40/Armadillo.ply"), "Armadillo99.9", 99.9, 50)
    process_model(Path("models/processed_ModelNet40/Armadillo.ply"), "Armadillo99.99", 99.98, 50)


if __name__ == "__main__":
    diff_num_points()
    #process_model(dragon_full, "dragon", 99, 10)

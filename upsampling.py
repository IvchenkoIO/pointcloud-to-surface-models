import open3d as o3d
import numpy as np
from pathlib import Path
import numpy as np
from plyfile import PlyData
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import upsampling_helper_funcs as up_h


def upsampling_alg(sparse_point_cloud,output_path,filename,extension):
    # Load sparse point cloud
    #sparse_pcd = o3d.io.read_point_cloud("models/bunny/bun_zipper_res4.ply")
    sparse_pcd = sparse_point_cloud
    print(f"Sparse: {len(sparse_pcd.points)} points")

    # Estimate normals
    sparse_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    sparse_pcd.orient_normals_consistent_tangent_plane(k=15)

    # Poisson reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        sparse_pcd, depth=9
    )

    # Sample dense points
    dense_pcd = mesh.sample_points_uniformly(
        number_of_points=len(sparse_pcd.points) * 10
    )

    print(f"Dense: {len(dense_pcd.points)} points")
    print(f"Points: {len(dense_pcd.points)} | Normals: {dense_pcd.has_normals()} | Colors: {dense_pcd.has_colors()}")
    # Visualize and save
    #o3d.visualization.draw_geometries([sparse_pcd])
    o3d.visualization.draw_geometries([dense_pcd])
    #o3d.io.write_point_cloud(str(output_path+filename+extension), dense_pcd)
    out = Path(output_path) / f"{filename}{extension}"
    out.parent.mkdir(parents=True, exist_ok=True)  # ensure folder exists
    o3d.io.write_point_cloud(str(out), dense_pcd)



def upsampling_self(filename,num_iterations=10000, k_neighbors=20, print_every=100):
    #num_iterations = 1000
    #k_neighbors = 20
    #print_every = 100



    sparse_points = up_h.data_preparation(filename=filename)
    dense_points = sparse_points.copy()
    print(f"\n{'=' * 60}")
    print(f"STARTING UPSAMPLING")
    print(f"{'=' * 60}")
    print(f"Initial points: {len(dense_points)}")
    print(f"Target: add {num_iterations} new points")
    print(f"{'-' * 60}")
    successful_adds = 0
    kdtree = up_h.kd_trees(sparse_points=dense_points)
    for iteration in range(num_iterations):

        #kdtree = up_h.kd_trees(sparse_points=dense_points)
        point_idx = np.random.randint(0, len(dense_points))

        u,v, neighbors,center = up_h.plane_fitting(kd_tree=kdtree,sparse_points=dense_points,point_idx=point_idx,k_neighbors=k_neighbors)

        points_2d = up_h.projection_3d_to_2d(u=u,v=v,neighbors=neighbors,center=center)

        try:
            vor = up_h.build_voronoi_diagram(points_2d=points_2d)
            best_vertex_2d, max_radius = up_h.finding_largest_gap(points_2d=points_2d, vor=vor)
        except Exception:
            continue

        if max_radius < 1e-9:
            continue

        new_point_3d = up_h.projection_2d_to_3d(best_vertex_2d, center, u, v)

        #local "jump" check to avoid bridging separate parts (bunny ears)
        base_point = dense_points[point_idx]

        #estimate local scale around the base point
        k_local = min(k_neighbors, len(dense_points))
        local_dists, _ = kdtree.query(base_point, k=k_local)
        #ignore the 0 distance to itself
        local_scale = np.median(local_dists[1:]) if k_local > 1 else local_dists[0]

        #distance from the base point to the new point
        jump_dist = np.linalg.norm(new_point_3d - base_point)

        #if the new point is far from the local neighborhood, skip
        if jump_dist > 2.0 * local_scale:
            continue

        nn_dist, _ = kdtree.query(new_point_3d, k=1)
        if nn_dist < 0.25 * max_radius:
            continue

        dense_points = np.vstack([dense_points, new_point_3d])
        successful_adds += 1

        # Progress update
        if (iteration + 1) % 20 == 0:
            kdtree = up_h.kd_trees(sparse_points=dense_points)

        if (iteration + 1) % print_every == 0:
            print(f"Iter {iteration + 1:4d}/{num_iterations}: "
                  f"{len(dense_points)} points (+{successful_adds}), gap={max_radius:.6f}")

    print(f"{'-' * 60}")
    print(f"COMPLETE!")
    print(f"Final: {len(dense_points)} points")
    print(f"Added: {successful_adds} new points")
    print(f"{'=' * 60}\n")
    dense_points = up_h.remove_sparse_outliers(dense_points, k=20, factor=2.5)
    print(f"After density filtering: {len(dense_points)} points")
    print(f"{'=' * 60}\n")
    up_h.save_ply("dense_output.ply", dense_points)
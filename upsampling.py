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
    #load sparse point cloud
    #sparse_pcd = o3d.io.read_point_cloud("models/bunny/bun_zipper_res4.ply")
    sparse_pcd = sparse_point_cloud
    print(f"Sparse: {len(sparse_pcd.points)} points")

    #estimate normals
    sparse_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    sparse_pcd.orient_normals_consistent_tangent_plane(k=15)

    #poisson reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        sparse_pcd, depth=9
    )

    #sample dense points
    dense_pcd = mesh.sample_points_uniformly(
        number_of_points=len(sparse_pcd.points) * 10
    )

    print(f"Dense: {len(dense_pcd.points)} points")
    print(f"Points: {len(dense_pcd.points)} | Normals: {dense_pcd.has_normals()} | Colors: {dense_pcd.has_colors()}")
    #visualize and save
    o3d.visualization.draw_geometries([dense_pcd])
    out = Path(output_path) / f"{filename}{extension}"
    out.parent.mkdir(parents=True, exist_ok=True)  #ensure folder exists
    o3d.io.write_point_cloud(str(out), dense_pcd)



def upsampling_self(model_name, filename,num_iterations=10000, k_neighbors=20, print_every=100):
    #own upsampling approach
    #load data
    sparse_points = up_h.data_preparation(filename=filename)
    dense_points = sparse_points.copy()

    #prints
    print(f"\n{'=' * 60}")
    print(f"STARTING UPSAMPLING")
    print(f"{'=' * 60}")
    print(f"Initial points: {len(dense_points)}")
    print(f"Target: add {num_iterations} new points")
    print(f"{'-' * 60}")


    successful_adds = 0
    #build KD-tree once, then periodically rebuild as points are added
    #this is much faster than rebuilding every loop iteration
    kdtree = up_h.kd_trees(sparse_points=dense_points)

    for iteration in range(num_iterations):
        #choose a random base point index to define a local neighborhood
        point_idx = np.random.randint(0, len(dense_points))

        u,v, neighbors,center = up_h.plane_fitting(kd_tree=kdtree,sparse_points=dense_points,point_idx=point_idx,k_neighbors=k_neighbors)

        points_2d = up_h.projection_3d_to_2d(u=u,v=v,neighbors=neighbors,center=center)

        #voronoi can fail for bad inputs
        #so we skip those iterations
        try:
            vor = up_h.build_voronoi_diagram(points_2d=points_2d)
            best_vertex_2d, max_radius = up_h.finding_largest_gap(points_2d=points_2d, vor=vor)
        except Exception:
            continue

        #if the "largest gap" is effectively zero, adding a point wont help
        if max_radius < 1e-9:
            continue

        #candidate point in 3D (lift from plane back to 3D).
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

        #if candidate is far beyond typical local spacing, it likely belongs
        #to a different sheet/part of the model -> reject
        if jump_dist > 2.0 * local_scale:
            continue

        #avoid inserting points too close to existing points (near-duplicates)
        nn_dist, _ = kdtree.query(new_point_3d, k=1)

        #if the candidate is already within a small fraction of the discovered gap radius
        #then the region is already filled
        if nn_dist < 0.25 * max_radius:
            continue

        #append to dense_points
        dense_points = np.vstack([dense_points, new_point_3d])
        successful_adds += 1

        #rebuild KD-tree occasionally so neighbor queries reflect inserted points
        if (iteration + 1) % 20 == 0:
            kdtree = up_h.kd_trees(sparse_points=dense_points)

        #progress logging
        if (iteration + 1) % print_every == 0:
            print(f"Iter {iteration + 1:4d}/{num_iterations}: "
                  f"{len(dense_points)} points (+{successful_adds}), gap={max_radius:.6f}")

    #prints
    print(f"{'-' * 60}")
    print(f"COMPLETE!")
    print(f"Final: {len(dense_points)} points")
    print(f"Added: {successful_adds} new points")
    print(f"{'=' * 60}\n")
    #remove points that are unusually isolated vs their neighbors
    dense_points = up_h.remove_sparse_outliers(dense_points, k=20, factor=2.5)
    print(f"After density filtering: {len(dense_points)} points")
    print(f"{'=' * 60}\n")
    #save
    up_h.save_ply(f"output/{model_name}/dense_output.ply", dense_points)
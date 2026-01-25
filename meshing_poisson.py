import open3d as o3d

def mesh_poisson(model_name):
    pcd = o3d.io.read_point_cloud(f"output/{model_name}/dense_output.ply")

    #checking oriented normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd.orient_normals_consistent_tangent_plane(k=50)

    #mesh from poisson
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=10, width=0, scale=1.1, linear_fit=False
    )

    #trim low-density (floating) parts
    import numpy as np
    dens = np.asarray(dens)
    keep = dens > np.quantile(dens, 0.05)   # drop bottom 5%
    mesh = mesh.select_by_index(np.where(keep)[0])

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # Add colors
    if pcd.has_colors():
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        mesh_vertices = np.asarray(mesh.vertices)
        pcd_colors = np.asarray(pcd.colors)

        colors = []
        for v in mesh_vertices:
            _, idx, _ = pcd_tree.search_knn_vector_3d(v, 1)
            colors.append(pcd_colors[idx[0]])

        mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(colors))
    else:
        # fallback: gray
        mesh.paint_uniform_color([0.7, 0.7, 0.7])

    o3d.io.write_triangle_mesh(f"output/{model_name}/mesh_poisson.ply", mesh)
    
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


if __name__ == "__main__":
    mesh_poisson("dragon99.9")

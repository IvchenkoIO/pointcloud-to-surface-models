import open3d as o3d
pcd = o3d.io.read_point_cloud("dense_output.ply")

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

o3d.io.write_triangle_mesh("mesh_poisson.ply", mesh)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
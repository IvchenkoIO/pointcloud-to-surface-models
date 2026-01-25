import open3d as o3d

mesh = o3d.io.read_triangle_mesh("output/chair/mesh_poisson.ply")

mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

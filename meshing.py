import open3d as o3d
import numpy as np
import pyvista as pv

#load data
pcd = o3d.io.read_point_cloud("dense_output.ply")
points = np.asarray(pcd.points)

cloud = pv.PolyData(points)

#Delaunay 3D with an alpha value
tet_mesh = cloud.delaunay_3d(alpha=0.02)

#extract just the outer surface of the tetra mesh
surface = tet_mesh.extract_surface()

# Visualize points + surface
plotter = pv.Plotter()
plotter.add_mesh(surface, show_edges=True, opacity=0.5)
plotter.add_mesh(cloud, render_points_as_spheres=True, point_size=4)
plotter.show()

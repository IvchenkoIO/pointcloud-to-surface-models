import open3d as o3d
import numpy as np
import pyvista as pv

def mesh_delaunay(model_name):
    #load data
    pcd = o3d.io.read_point_cloud(f"output/{model_name}/dense_output.ply")
    points = np.asarray(pcd.points)

    cloud = pv.PolyData(points)

    #Delaunay 3D with an alpha value
    tet_mesh = cloud.delaunay_3d(alpha=30)

    #extract just the outer surface of the tetra mesh
    surface = tet_mesh.extract_surface()
    
    # surface.save(f"output/{model_name}/mesh_delaunay.ply")

    # Visualize points + surface
    plotter = pv.Plotter()
    plotter.add_mesh(surface, show_edges=True, opacity=0.5)
    plotter.add_mesh(cloud, render_points_as_spheres=True, point_size=4)
    plotter.show()


if __name__ == "__main__":
    mesh_delaunay("airplane")

import os
from pointcloud_to_surface_models import process_model
from meshing_poisson import mesh_poisson
from meshing_delaunay import mesh_delaunay


folder = "models/processed_ModelNet40"

for filename in os.listdir(folder):
    filepath = os.path.join(folder, filename)
    model_name = os.path.splitext(filename)[0]

    if os.path.isfile(filepath):
        print(filename)
        process_model(filepath, model_name, 99, 10)
        mesh_poisson(model_name)
        mesh_delaunay(model_name)

import open3d as o3d
from pathlib import Path
import numpy as np
import random


# -------------------------
# CONFIG
# -------------------------
MODELNET40_ROOT = Path("models/ModelNet40")       # folder containing the categories
OUTPUT_DIR = Path("models/processed_ModelNet40") # where .ply clouds will be saved
POINTS_PER_MODEL = 200_000                 # number of sampled points on mesh
N_PER_CATEGORY = 10                        # <--- choose how many per class

RANDOM_SEED = 42                           # reproducibility
random.seed(RANDOM_SEED)


# -------------------------
# HELPERS
# -------------------------

def load_mesh(filepath):
    """Loads .off file as Open3D TriangleMesh."""
    mesh = o3d.io.read_triangle_mesh(str(filepath))
    if mesh.is_empty():
        print(f"Empty mesh: {filepath}")
        return None
    mesh.compute_vertex_normals()
    return mesh


def sample_point_cloud(mesh, n_points):
    """Uniformly sample dense point cloud from mesh."""
    return mesh.sample_points_uniformly(number_of_points=n_points)


def save_pcd(pcd, out_path):
    """Save point cloud to PLY."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out_path), pcd)


# -------------------------
# MAIN PROCESSING
# -------------------------

def preprocess_random_subset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    categories = sorted([d for d in MODELNET40_ROOT.iterdir() if d.is_dir()])

    print(f"Found {len(categories)} categories.")

    for cat_dir in categories:
        class_name = cat_dir.name
        print(f"\n=== Category: {class_name} ===")

        # Collect all .off files from both train and test
        off_files = list((cat_dir / "train").glob("*.off")) + \
                    list((cat_dir / "test").glob("*.off"))

        if len(off_files) == 0:
            print(f"No .off files for category {class_name}")
            continue

        # Shuffle and pick N models
        random.shuffle(off_files)
        selected = off_files[:N_PER_CATEGORY]

        print(f"Selected {len(selected)} models out of {len(off_files)} total.")

        # Process selected models
        for off_file in selected:
            print(f"Processing {off_file.name}")

            mesh = load_mesh(off_file)
            if mesh is None:
                continue

            pcd = sample_point_cloud(mesh, POINTS_PER_MODEL)

            out_name = off_file.stem + "_dense.ply"
            out_path = OUTPUT_DIR / class_name / out_name

            save_pcd(pcd, out_path)

    print("\nDone. Preprocessed point clouds saved to:")
    print(f"  {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    preprocess_random_subset()
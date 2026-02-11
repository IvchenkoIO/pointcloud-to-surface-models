# Pointcloud → Surface Models

This project turns a **sparse point cloud** into a **denser point cloud** (custom Voronoi-gap upsampling), then reconstructs a **surface mesh** using:
- **Poisson reconstruction** (Open3D)
- **Delaunay 3D (alpha shape)** (PyVista)

---

## 1) Requirements

### OS / hardware
- Works on Windows / macOS / Linux.
- **Visualization windows** are opened (Open3D / PyVista). You need a machine with a GUI (not headless).

### Python
- Python 3.9+ recommended.

### Python dependencies
Install these packages (match the imports across the scripts):

```bash
pip install numpy scipy matplotlib plyfile open3d pyvista
```

---

## 2) Scripts (what each file does)

- `pointcloud_to_surface_models.py` — core pipeline:
  - loads a dense “ground truth” point cloud
  - randomly removes points to make it sparse
  - runs the custom upsampler to create a dense reconstruction
  - optionally evaluates dense vs ground truth via `comparison.evaluate(...)`
- `upsampling.py` + `upsampling_helper_funcs.py` — custom upsampling algorithm (local plane fit + 2D Voronoi “largest gap” insertion + filtering).
- `meshing_poisson.py` — Poisson surface reconstruction from the dense point cloud and saves `mesh_poisson.ply`.
- `meshing_delaunay.py` — Delaunay 3D / alpha surface extraction (PyVista). Currently **visualizes** but does **not save** the mesh (save line is commented).
- `process_and_save_all.py` — batch runner over files in `models/processed_ModelNet40`.
- `preprocessing_ModelNet40.py` — helper to sample dense point clouds from ModelNet40 `.off` meshes into `.ply` point clouds.
- `visualize_point_cloud.py` / `visualize_poisson_mesh.py` — small viewers for point clouds / Poisson meshes.

---

## 3) Setup 

### Create and activate a virtual environment

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```
> **Note:** If you get a PowerShell execution policy error, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```


### Install dependencies
```bash
pip install numpy scipy matplotlib plyfile open3d pyvista
```

⚠️ **Important path note (batch runner):**  
While you can just run the files directly (for example opening them in PyCharm IDE), we still recommend using virtual env in case you do not want to download dependencies globally

---

## 4) Prepare input data

The pipeline expects a **point cloud file** (PLY) as “ground truth”, then it creates a sparse version by randomly removing points.

### Option A (recommended): Use ModelNet40 + preprocessing script

1) Download ModelNet40 and place it at:

```
models/ModelNet40/
  airplane/
    train/*.off
    test/*.off
  chair/
    train/*.off
    test/*.off
  ...
```

2) Run preprocessing (samples `POINTS_PER_MODEL` points from each mesh and writes `.ply` files):
```bash
python preprocessing_ModelNet40.py
```

This writes into:
```
models/processed_ModelNet40/<category>/<model>_dense.ply
```

⚠️ **Important path note (batch runner):**  
`process_and_save_all.py` lists only files directly inside `models/processed_ModelNet40` (non-recursive).  
If your preprocessing output is nested by category (as above), either:
- **Copy** one or more `.ply` files into `models/processed_ModelNet40/` (flat), **or**
- Modify `process_and_save_all.py` to walk subdirectories (e.g., `os.walk`).

### Option B: Use your own `.ply` point cloud
Put a dense point cloud `.ply` anywhere (e.g., `models/custom/my_model.ply`). Then in `pointcloud_to_surface_models.py` hardcode your cloud path.

---

## 5) Run the pipeline

### A) Run a single model (best for grading / quick test)

This runs:
1) create sparse point cloud
2) upsample (custom method)
3) Poisson mesh
4) Delaunay surface (visual)

```bash
python -c "from pointcloud_to_surface_models import process_model; from meshing_poisson import mesh_poisson; from meshing_delaunay import mesh_delaunay; process_model('models/processed_ModelNet40/example.ply','example',99,10); mesh_poisson('example'); mesh_delaunay('example')"
```

Where:
- `99` = percent of points removed (so 1% kept)
- `10` = upscale factor (target is ~10× more points than sparse)

Notes:
- Visualization windows will appear; close each window to continue.

### B) Batch run (process all files in a folder)

Put multiple `.ply` files directly inside:
```
models/processed_ModelNet40/
```
Then run:
```bash
python process_and_save_all.py
```

### C) Run a deep-learning version

We also provide a `Deep_Learning.ipynb` notebook, to run it -> just open it in the Google Collab environment.
---

## 6) Outputs

For each `model_name`, the pipeline writes:
```
output/<model_name>/
  sparse_output.ply
  dense_output.ply
  mesh_poisson.ply
```

**Delaunay output:** currently visualized but not saved because the save line is commented.  
To save it, uncomment:
```python
# surface.save(f"output/{model_name}/mesh_delaunay.ply")
```

---

## 7) Visualization helpers

- View a dense point cloud:
  - edit the path inside `visualize_point_cloud.py` and run:
```bash
python visualize_point_cloud.py
```

- View a Poisson mesh:
  - edit the path inside `visualize_poisson_mesh.py` and run:
```bash
python visualize_poisson_mesh.py
```

---

## 8) Troubleshooting / common issues

### `ModuleNotFoundError: comparison`
`pointcloud_to_surface_models.py` imports `evaluate` from `comparison`.  
If `comparison.py` is not present in your submission folder, either add it or comment out the evaluation call:
```python
# from comparison import evaluate
# evaluate(full_cloud, dense, model_name)
```

### Headless environments
Open3D / PyVista visualization requires a display. Run on a machine with a desktop environment.

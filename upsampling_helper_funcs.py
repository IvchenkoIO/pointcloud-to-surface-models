import open3d as o3d
import numpy as np
from pathlib import Path
import numpy as np
from plyfile import PlyData,PlyElement
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from numpy.linalg import norm

def data_preparation(filename):
    print("-----------------")
    print("DATA PREPARATION")
    print("-----------------")
    plydata = PlyData.read(filename)
    vertex = plydata['vertex']
    sparse_points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    print(f"Loaded {len(sparse_points)} points")
    print(f"Point cloud bounds: {sparse_points.min(axis=0)} to {sparse_points.max(axis=0)}")
    return sparse_points

def kd_trees(sparse_points):
    #print("-----------------")
    #print("Building KDTrees")
    #print("-----------------")
    kdtree = KDTree(sparse_points)
    return kdtree

def plane_fitting(kd_tree,sparse_points,point_idx,k_neighbors):
    distances, neighbor_indices = kd_tree.query(sparse_points[point_idx], k=k_neighbors)
    neighbors = sparse_points[neighbor_indices]

    # weighted center (MLS-ish, simple Gaussian weight)
    sigma = max(np.median(distances), 1e-12)
    w = np.exp(-(distances ** 2) / (2 * sigma ** 2))
    w /= (w.sum() + 1e-12)
    center = (neighbors * w[:, None]).sum(axis=0)

    centered = neighbors - center
    C = (centered * w[:, None]).T @ centered  # weighted covariance
    evals, evecs = np.linalg.eigh(C)

    # sort largest -> smallest
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]

    u = evecs[:, 0];
    v = evecs[:, 1]

    # defensive orthonormalization
    u = u / (norm(u) + 1e-12)
    v = v - u * (u @ v)
    v = v / (norm(v) + 1e-12)

    return u, v, neighbors, center

def projection_3d_to_2d(u,v,neighbors, center):
    #print("-----------------")
    #print("3D to 2D projection")
    #print("-----------------")
    centered = neighbors - center
    # Project onto basis vectors
    x_2d = np.dot(centered, u)  # Coordinate along u direction
    y_2d = np.dot(centered, v)  # Coordinate along v direction
    points_2d = np.column_stack([x_2d, y_2d])
    #print(f"2D projected points shape: {points_2d.shape}")
    #print(f"First few 2D points:\n{points_2d[:3]}")
    return points_2d

def projection_2d_to_3d(point_2d, center, u, v):
    #print("\n-----------------")
    #print("PROJECTING BACK TO 3D")
    #print("-----------------")
    point_3d = center + point_2d[0] * u + point_2d[1] * v
    return point_3d

def build_voronoi_diagram(points_2d):
    #print("\n-----------------")
    #print("VORONOI DIAGRAM")
    #print("-----------------")

    # Compute Voronoi diagram in 2D
    vor = Voronoi(points_2d)

    #print(f"Number of Voronoi vertices: {len(vor.vertices)}")
    #print(f"Number of Voronoi regions: {len(vor.regions)}")
    return vor

def finding_largest_gap(points_2d, vor):
    ##1st approach to rmax
    #radii = norm(points_2d, axis=1)
    #r = np.linalg.norm(points_2d, axis=1)
    #r_med = np.median(radii) if len(radii) else 0.0
    #Rmax = 1.5 * r_med + 1e-9

    #2nd approach to rmax
    r = np.linalg.norm(points_2d, axis=1)
    r_med = np.median(r)
    r_iqr = np.subtract(*np.percentile(r, [75, 25])) + 1e-12
    Rmax = r_med + 1.5 * r_iqr  # robust instead of fixed 1.5×median


    max_radius = 0.0
    best_vertex_2d = None

    # Only finite vertices within a reasonable radius
    for vertex in vor.vertices:
        if not np.all(np.isfinite(vertex)):
            continue
        if norm(vertex) > Rmax:
            continue

        dmin = norm(points_2d - vertex, axis=1).min()
        if dmin > max_radius:
            max_radius = dmin
            best_vertex_2d = vertex

    return best_vertex_2d, float(max_radius)

def save_ply(filename, points, ascii=True):
    """Save points to PLY file"""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)

    # build structured array
    vertex = np.empty(pts.shape[0],
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex['x'] = pts[:, 0]
    vertex['y'] = pts[:, 1]
    vertex['z'] = pts[:, 2]

    el = PlyElement.describe(vertex, 'vertex')

    # choose ASCII vs binary here
    ply = PlyData([el], text=ascii)
    ply.write(filename)
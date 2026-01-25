import numpy as np
from scipy.spatial import cKDTree

def chamfer_distance(A, B):
    treeA = cKDTree(A)
    treeB = cKDTree(B)
    distA, _ = treeA.query(B)
    distB, _ = treeB.query(A)
    return np.mean(distA**2) + np.mean(distB**2)

def estimate_offset_direction(points):
    cov = np.cov(points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    normal = eigenvectors[:, np.argmin(eigenvalues)]
    return normal / np.linalg.norm(normal)

def offset_direction_length(points, normal):
    projections = points @ normal
    return projections.max() - projections.min()

def normalized_chamfer(A, cd):
    points = np.asarray(A)
    normal = estimate_offset_direction(points)
    L = offset_direction_length(points, normal)

    CD_norm = cd / (L * L)
    return CD_norm

def hausdorff_distance(A, B):
    treeA = cKDTree(A)
    treeB = cKDTree(B)
    distA, _ = treeA.query(B)
    distB, _ = treeB.query(A)
    return max(np.max(distA), np.max(distB))

def evaluate(ground_truth, reconstructed, model_name):
    cd = chamfer_distance(ground_truth.points, reconstructed.points)
    cd_norm = normalized_chamfer(ground_truth.points, cd)
    hd = hausdorff_distance(ground_truth.points, reconstructed.points)

    text = (
        f"Chamfer Distance: {cd}\n"
        "Normalized Chamfer Distance: 0 - exact match. 1 and more - bad match.\n"
        f"Normalized Chamfer Distance: {cd_norm}\n"
        f"Hausdorff Distance: {hd}\n"
    )
    
    with open(f"output/{model_name}/comparison.txt", "w") as f:
        f.write(text)

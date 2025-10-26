from open3d import io, geometry, visualization
import numpy as np


armadillo = "models/armadillo/Armadillo.ply" # remove 90+%

buddha_full = "models/buddha/happy_vrip.ply" # remove 95+%
buddha_small = "models/buddha/happy_vrip_res2.ply"
buddha_smaller = "models/buddha/happy_vrip_res3.ply"
buddha_smallest = "models/buddha/happy_vrip_res4.ply" # maybe remove like 50% to add some randomness

bunny_full = "models/bunny/bun_zipper.ply" # remove 80+%
bunny_small = "models/bunny/bun_zipper_res2.ply"
bunny_smaller = "models/bunny/bun_zipper_res3.ply"
bunny_smallest = "models/bunny/bun_zipper_res4.ply" # good as is

dragon_full = "models/dragon/dragon_vrip.ply" # remove 95+%
dragon_small = "models/dragon/dragon_vrip_res2.ply"
dragon_smaller = "models/dragon/dragon_vrip_res3.ply"
dragon_smallest = "models/dragon/dragon_vrip_res4.ply" # maybe remove like 50% to add some randomness

lucy = "models/lucy/lucy.ply" # It's extremely large, remove like 99.9% points


def load_and_reduce_point_cloud(filename: str, percentage: float):
    cloud = io.read_point_cloud(filename)
    num_points = len(cloud.points)

    keep_ratio = 1 - (percentage / 100)
    num_keep = int(num_points * keep_ratio)

    indices = np.random.choice(num_points, num_keep, replace=False)
    reduced_cloud = cloud.select_by_index(indices)

    return reduced_cloud


def visualize_point_cloud(cloud):
    visualization.draw_geometries([cloud])


def main():
    filename = bunny_smallest
    percentage_to_remove = 0
    reduced_cloud = load_and_reduce_point_cloud(filename, percentage_to_remove)

    visualize_point_cloud(reduced_cloud)


if __name__ == "__main__":
    main()

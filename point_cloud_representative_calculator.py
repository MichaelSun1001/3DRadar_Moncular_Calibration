import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.cluster import KMeans


def save_point_cloud_data_to_csv(data, csv_filename):
    """Save representative point data to a CSV file"""
    df = pd.DataFrame(data)
    df.to_csv(csv_filename, index=False)


def plot_point_cloud_2d(frame_data, output_path, frame_number):
    """Plot and save 2D point cloud for the current frame"""
    fig, ax = plt.subplots()
    xs = frame_data["X"]
    ys = frame_data["Y"]
    snrs = frame_data["SNR"]
    indices = frame_data.index

    scatter = ax.scatter(xs, ys, c=snrs, cmap="viridis", marker="o")
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.text(x, y, f"{indices[i]}", fontsize=9)

    plt.colorbar(scatter, ax=ax, label="SNR")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title(f"Point Cloud Data for Frame {frame_number}")

    frame_filename = f"frame_{frame_number}_point_cloud.png"
    frame_filepath = os.path.join(output_path, frame_filename)
    plt.savefig(frame_filepath)
    plt.close(fig)


def find_representative_point_weighted(frame_data):
    """Calculate representative point (X,Y) using weighted average by SNR"""
    coords = frame_data[["X", "Y"]].values
    snrs = frame_data["SNR"].values
    if len(coords) == 0:
        return None
    weights = snrs / snrs.sum()
    return np.average(coords, axis=0, weights=weights)


def find_representative_point_distances(frame_data):
    """Calculate representative point (X,Y) based on pairwise distances"""
    coords = frame_data[["X", "Y"]].values
    num = len(coords)
    if num < 3:
        return None
    dist_matrix = distance.cdist(coords, coords, "euclidean")
    np.fill_diagonal(dist_matrix, np.inf)
    num_closest = int(num * 4 / 5)
    flat_idxs = dist_matrix.argsort(axis=None)[:num_closest]
    rows, _ = np.unravel_index(flat_idxs, dist_matrix.shape)
    selected = coords[rows]
    return selected.mean(axis=0)


def find_representative_point_kmeans(frame_data):
    """Calculate representative point (X,Y) using K-means centroid"""
    coords = frame_data[["X", "Y"]].values
    if len(coords) == 0:
        return None
    kmeans = KMeans(n_clusters=1, random_state=0).fit(coords)
    return kmeans.cluster_centers_[0]


def main(input_csv, output_directory, method="weighted_average"):
    """Process 2D point cloud data and save outputs"""
    os.makedirs(output_directory, exist_ok=True)
    df = pd.read_csv(input_csv, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    frames = df.groupby("Frame")
    rep_points = []

    for frame, frame_data in frames:
        frame_data = frame_data.reset_index(drop=True)
        plot_point_cloud_2d(frame_data, output_directory, frame)
        if method == "weighted_average":
            rep = find_representative_point_weighted(frame_data)
        elif method == "distances":
            rep = find_representative_point_distances(frame_data)
        elif method == "kmeans":
            rep = find_representative_point_kmeans(frame_data)
        else:
            raise ValueError(f"Unknown method: {method}")
        if rep is not None:
            rep_points.append({"Frame": frame, "Rep_X": rep[0], "Rep_Y": rep[1]})
        else:
            print(f"Frame {frame} does not have enough points.")

    rep_csv = os.path.join(output_directory, "representative_points.csv")
    save_point_cloud_data_to_csv(rep_points, rep_csv)


if __name__ == "__main__":
    input_csv = r"C:\Users\10137\Desktop\3DRadar_Moncular_Calibration\dataset\pointcloud\csv\pointcloud.csv"
    output_dir = (
        r"C:\Users\10137\Desktop\3DRadar_Moncular_Calibration\dataset\pointcloud\csv"
    )
    method = "distances"  # 'weighted_average', 'distances', or 'kmeans'
    main(input_csv, output_dir, method)


# 加权平均法(weighted_average)：
# 适用情况：点云的信噪比(SNR) 数据准确，并且 SNR 能够反映点的权重（即高 SNR 表示更重要的点）。
# 优点：能够利用 SNR 信息，使得计算的代表点更具有物理意义。
# 缺点：如果 SNR 数据不准确或不稳定，结果可能会受到影响。

# 距离最小法(distances)：
# 适用情况：点云数据中存在较多的噪声，但总体上点分布较为集中。
# 优点：通过选择距离最小的 n*3/4 个点，可以减少噪声点对代表点的影响。
# 缺点：如果点云数据本身非常分散或者有多个聚类中心，这种方法可能不太适合。

# K-means 聚类法(kmeans)：
# 适用情况：点云数据具有明显的聚类特性，并且希望通过聚类算法找到中心点。
# 优点：能够有效处理聚类特性的点云数据，并找到聚类中心作为代表点。
# 缺点：对噪声敏感，且需要设定聚类数量（这里设定为 1，找到单个中心点）。

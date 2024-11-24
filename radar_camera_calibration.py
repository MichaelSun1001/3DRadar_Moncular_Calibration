import cv2
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def cost_function(params, radar_points, camera_points, camera_matrix):
    """
    Define the cost function for optimization.

    Parameters:
    params (np.array): Array containing rotation and translation parameters
    radar_points (np.array): Radar point cloud data
    camera_points (np.array): Pixel points in the camera image
    camera_matrix (np.array): Camera intrinsic matrix

    Returns:
    np.array: Error between projected points and camera pixel points
    """
    # Extract rotation vector and translation vector
    rotation_vector = params[:3]
    translation_vector = params[3:]

    # Convert rotation vector to rotation matrix
    rotation_matrix = cv2.Rodrigues(rotation_vector)[0]

    # Compute radar points projected into camera coordinate system
    radar_points_homogeneous = np.hstack(
        (radar_points, np.ones((radar_points.shape[0], 1))))
    projected_points_homogeneous = np.dot(radar_points_homogeneous, np.hstack(
        (rotation_matrix, translation_vector.reshape(-1, 1))).T)
    projected_points_homogeneous = np.dot(
        projected_points_homogeneous, camera_matrix.T)
    projected_points = projected_points_homogeneous[:,
                                                    :2] / projected_points_homogeneous[:, 2:]

    # Calculate error between predicted points and actual points
    return (projected_points - camera_points).flatten()


def calibrate_extrinsics(radar_points, camera_points, camera_matrix):
    """
    Calibrate extrinsic parameters using RANSAC and optimization.

    Parameters:
    radar_points (np.array): Radar point cloud data
    camera_points (np.array): Pixel points in the camera image
    camera_matrix (np.array): Camera intrinsic matrix

    Returns:
    tuple: Optimized rotation matrix, translation vector
    """
    # Ensure we have at least 4 points
    if radar_points.shape[0] < 4 or camera_points.shape[0] < 4:
        raise ValueError("At least 4 point pairs are required for PnP RANSAC.")

    # Ensure data types are correct
    if radar_points.dtype != np.float32 or camera_points.dtype != np.float32:
        raise ValueError(
            "Radar points and camera points must be of type float32.")

    # Use solvePnPRansac for initial estimation
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        radar_points, camera_points, camera_matrix, distCoeffs=None, flags=cv2.SOLVEPNP_ITERATIVE)

    if not success:
        raise Exception("PnP RANSAC failed to find a valid solution")

    initial_params = np.hstack((rvec.flatten(), tvec.flatten()))

    # Use Levenberg-Marquardt algorithm for refinement
    result = least_squares(
        cost_function,
        initial_params,
        args=(radar_points, camera_points, camera_matrix),
        method='lm',
        ftol=1e-8,    # Function tolerance
        xtol=1e-8,    # Parameter vector tolerance
        gtol=1e-8,    # Gradient norm tolerance
        max_nfev=500  # Maximum number of function evaluations
    )

    if not result.success:
        raise Exception("Levenberg-Marquardt optimization failed")

    optimized_params = result.x
    rotation_matrix = cv2.Rodrigues(optimized_params[:3])[0]
    translation_vector = optimized_params[3:]

    return rotation_matrix, translation_vector


def load_radar_points(file_path):
    """
    Load radar points from CSV file.

    Parameters:
    file_path (str): Path to the CSV file

    Returns:
    np.array: Radar point cloud data
    """
    radar_data = pd.read_csv(file_path)
    radar_points = radar_data[['Rep_X', 'Rep_Y',
                               'Rep_Z']].values.astype(np.float32)
    print("Radar points loaded:")
    print_data_points(radar_points)
    return radar_points


def load_camera_points(file_path):
    """
    Load camera points from CSV file.

    Parameters:
    file_path (str): Path to the CSV file

    Returns:
    np.array: Pixel points in the camera image
    """
    camera_data = pd.read_csv(file_path)
    camera_points = camera_data[['X', 'Y']].values.astype(np.float32)
    print("Camera points loaded:")
    print_data_points(camera_points)
    return camera_points


def print_data_points(points):
    """
    Print the first and last three data points with ellipsis in the middle.

    Parameters:
    points (np.array): Array of data points
    """
    num_points = points.shape[0]
    if num_points > 6:
        points_to_print = np.vstack(
            [points[:3], [["..."] * points.shape[1]], points[-3:]])
    else:
        points_to_print = points

    for point in points_to_print:
        formatted_point = "[" + " ".join(f"{p:<12}" for p in point) + "]"
        print(formatted_point)


def main():
    # Define camera intrinsic matrix (example values, needs actual calibration)
    fx, fy, cx, cy = 957.01226348, 934.81781967, 982.79102164, 499.9173914
    camera_matrix = np.array(
        [[fx, 0, cx],
         [0, fy, cy],
         [0, 0, 1]], dtype=np.float32)

    # Directly specify the paths to the CSV files
    radar_csv_path = r'C:\Users\sunaxiang\Desktop\Radar_Moncular_Calibration\4DRadar_Moncular_Calibration\dataset\pointcloud\csv\representative_points.csv'
    camera_csv_path = r'C:\Users\sunaxiang\Desktop\Radar_Moncular_Calibration\4DRadar_Moncular_Calibration\dataset\image\csv\camera_incenter.csv'

    # Load radar point cloud and camera pixel point pairs from CSV files
    radar_points = load_radar_points(radar_csv_path)
    camera_points = load_camera_points(camera_csv_path)

    # Check if the number of radar points matches the number of camera points
    if radar_points.shape[0] != camera_points.shape[0]:
        raise ValueError(
            "Number of radar points does not match number of camera points")

    # Calibrate extrinsic parameters
    try:
        rotation_matrix, translation_vector = calibrate_extrinsics(
            radar_points, camera_points, camera_matrix)

        print("Optimized rotation matrix:")
        print(rotation_matrix)
        print("Optimized translation vector:")
        print(translation_vector)
    except Exception as e:
        print(f"Calibration failed: {e}")


if __name__ == "__main__":
    main()

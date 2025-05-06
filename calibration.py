# import numpy as np
# import pandas as pd
# import cv2


# def main():
#     radar_csv = r"C:\Users\10137\Desktop\3DRadar_Moncular_Calibration\dataset\pointcloud\csv\representative_points.csv"
#     pixel_csv = r"C:\Users\10137\Desktop\3DRadar_Moncular_Calibration\dataset\image\csv\camera_incenter.csv"
#     Homography_txt = (
#         r"C:\Users\10137\Desktop\3DRadar_Moncular_Calibration\dataset\Homography.txt"
#     )
#     radar_df = pd.read_csv(radar_csv)
#     pixel_df = pd.read_csv(pixel_csv)
#     merged = pd.merge(radar_df, pixel_df, on="Frame")
#     src = merged[["Rep_X", "Rep_Y"]].to_numpy(dtype=np.float32)
#     dst = merged[["X", "Y"]].to_numpy(dtype=np.float32)
#     H, mask = cv2.findHomography(src, dst, cv2.RANSAC)
#     np.savetxt(Homography_txt, H, fmt="%.6f")
#     print(H)


# if __name__ == "__main__":
#     main()

import numpy as np
import pandas as pd
import cv2


def refine_homography(src, dst, H0, threshold=0.1, max_iter=1000, eps=1e-6):
    H = H0
    for _ in range(max_iter):
        ones = np.ones((src.shape[0], 1))
        pts_hom = np.hstack((src, ones))
        proj = (H.dot(pts_hom.T)).T
        proj_xy = proj[:, :2] / proj[:, 2:3]
        errors = np.linalg.norm(proj_xy - dst, axis=1)
        inliers = errors < threshold
        if inliers.sum() < 4:
            break
        H_new, _ = cv2.findHomography(src[inliers], dst[inliers], 0)
        if np.linalg.norm(H_new - H) < eps:
            H = H_new
            break
        H = H_new
    return H


def main():
    radar_csv = r"C:\Users\10137\Desktop\3DRadar_Moncular_Calibration\dataset\pointcloud\csv\representative_points.csv"
    pixel_csv = r"C:\Users\10137\Desktop\3DRadar_Moncular_Calibration\dataset\image\csv\camera_incenter.csv"
    Homography_txt = (
        r"C:\Users\10137\Desktop\3DRadar_Moncular_Calibration\dataset\Homography.txt"
    )
    radar_df = pd.read_csv(radar_csv)
    pixel_df = pd.read_csv(pixel_csv)
    merged = pd.merge(radar_df, pixel_df, on="Frame")
    src = merged[["Rep_X", "Rep_Y"]].to_numpy(dtype=np.float32)
    dst = merged[["X", "Y"]].to_numpy(dtype=np.float32)
    H0, mask = cv2.findHomography(src, dst, cv2.RANSAC)
    H = refine_homography(src, dst, H0)
    np.savetxt(Homography_txt, H, fmt="%.6f")
    print(H)


if __name__ == "__main__":
    main()

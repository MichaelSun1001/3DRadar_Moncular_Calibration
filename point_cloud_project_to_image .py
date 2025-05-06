import numpy as np
import pandas as pd
import os
import cv2


def project_with_homography(radar_points, homography):
    ones = np.ones((radar_points.shape[0], 1))
    pts_hom = np.hstack([radar_points, ones])
    img_hom = (homography @ pts_hom.T).T
    img_pts = img_hom[:, :2] / img_hom[:, 2:3]
    return img_pts


def main():
    radar_file = r"C:\Users\10137\Desktop\3DRadar_Moncular_Calibration\dataset\pointcloud\csv\representative_points.csv"
    pixel_file = r"C:\Users\10137\Desktop\3DRadar_Moncular_Calibration\dataset\image\csv\camera_incenter.csv"
    image_dir = r"C:\Users\10137\Desktop\3DRadar_Moncular_Calibration\dataset\image"
    out_dir = r"C:\Users\10137\Desktop\3DRadar_Moncular_Calibration\dataset\calib"
    homography_txt = (
        r"C:\Users\10137\Desktop\3DRadar_Moncular_Calibration\dataset\Homography.txt"
    )
    os.makedirs(out_dir, exist_ok=True)
    csv_out = os.path.join(out_dir, "projected_points.csv")

    homography_matrix = np.loadtxt(homography_txt, dtype=np.float64)
    radar_df = pd.read_csv(radar_file)
    pixel_df = pd.read_csv(pixel_file)

    output_data = []

    for frame in radar_df["Frame"].unique():
        pts_radar = radar_df[radar_df["Frame"] == frame][["Rep_X", "Rep_Y"]].to_numpy()
        pts_pixel = pixel_df[pixel_df["Frame"] == frame]
        img_pts = project_with_homography(pts_radar, homography_matrix)
        img_path = os.path.join(image_dir, f"frame_{frame}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: 图像 {img_path} 未找到，跳过 frame {frame}")
            continue
        for (px, py), (ux, uy) in zip(pts_pixel[["X", "Y"]].values, img_pts):
            cv2.circle(img, (int(px), int(py)), 5, (255, 0, 0), -1)
            cv2.putText(
                img,
                f"P({px},{py})",
                (int(px) + 5, int(py) + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )
            cv2.circle(img, (int(ux), int(uy)), 5, (0, 0, 255), -1)
            cv2.putText(
                img,
                f"R({int(ux)},{int(uy)})",
                (int(ux) + 5, int(uy) + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )
            cv2.line(img, (int(px), int(py)), (int(ux), int(uy)), (0, 255, 0), 2)
            output_data.append([frame, px, py, float(ux), float(uy)])
        out_img = os.path.join(out_dir, f"annotated_frame_{frame}.jpg")
        cv2.imwrite(out_img, img)

    out_df = pd.DataFrame(
        output_data, columns=["Frame", "Pixel_X", "Pixel_Y", "Proj_X", "Proj_Y"]
    )
    out_df.to_csv(csv_out, index=False)
    print(f"投影结果已保存到 {csv_out}")


if __name__ == "__main__":
    main()

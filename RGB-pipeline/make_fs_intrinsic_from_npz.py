# make_fs_intrinsic_from_npz.py
import numpy as np
import os

CALIB_FILE = "stereo_calib_rgb.npz"            # 你的 npz
FS_ASSETS_DIR = "/home/match/FS/FoundationStereo/assets"
OUT_K_PATH = os.path.join(FS_ASSETS_DIR, "K_d435_2RGB.txt")

def main():
    calib = np.load(CALIB_FILE)
    P1 = calib["P1"]       # 3x4
    T = calib["T"].reshape(3, 1)

    # rectified 左相机内参: 取 P1 的前 3x3
    K_rect = P1[:, :3]     # 3x3
    fx = K_rect[0, 0]
    fy = K_rect[1, 1]
    cx = K_rect[0, 2]
    cy = K_rect[1, 2]

    # 基线 (米)
    baseline = float(np.linalg.norm(T))

    # 展平成 1x9
    K_flat = K_rect.reshape(-1)  # 按行展开: fx, 0, cx, 0, fy, cy, 0, 0, 1

    os.makedirs(FS_ASSETS_DIR, exist_ok=True)
    with open(OUT_K_PATH, "w") as f:
        f.write(" ".join(map(str, K_flat.tolist())) + "\n")
        f.write(str(baseline) + "\n")

    print("写出 FS 内参文件到:", OUT_K_PATH)
    print("K_rect =\n", K_rect)
    print("baseline (m) =", baseline)

if __name__ == "__main__":
    main()

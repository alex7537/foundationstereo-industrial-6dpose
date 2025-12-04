import numpy as np
import cv2
import json
import os
import open3d as o3d

out_dir = "FoundationStereo/shared_fs_test"   # 和之前的一样
fs_disp_path = os.path.join(out_dir, "depth_meter.npy")  # 你的 FS 输出

# =========================================================
# 1. 函数：IR 深度 → 对齐到 RGB + 点云
# =========================================================
def project_ir_depth_to_color(
    depth_ir_m: np.ndarray,
    K_ir: dict,
    K_color: dict,
    R_ir2color: np.ndarray,
    t_ir2color: np.ndarray,
    color_img_bgr: np.ndarray
):
    H_ir, W_ir = depth_ir_m.shape
    assert H_ir == K_ir["height"] and W_ir == K_ir["width"], "IR depth 尺寸要和 IR 内参匹配"

    vs, us = np.meshgrid(np.arange(H_ir), np.arange(W_ir), indexing="ij")
    Z_ir = depth_ir_m
    valid = Z_ir > 0

    if not np.any(valid):
        print("Warning: no valid depth in IR depth map.")
        return None, None, None

    fx_ir, fy_ir = K_ir["fx"], K_ir["fy"]
    cx_ir, cy_ir = K_ir["cx"], K_ir["cy"]

    X_ir = (us - cx_ir) * Z_ir / fx_ir
    Y_ir = (vs - cy_ir) * Z_ir / fy_ir

    X_ir = X_ir[valid]
    Y_ir = Y_ir[valid]
    Z_ir = Z_ir[valid]

    pts_ir = np.stack([X_ir, Y_ir, Z_ir], axis=1)  # N x 3

    # IR → Color
    pts_ir_T = pts_ir.T  # 3 x N
    pts_color = (R_ir2color @ pts_ir_T + t_ir2color).T  # N x 3
    X_c = pts_color[:, 0]
    Y_c = pts_color[:, 1]
    Z_c = pts_color[:, 2]

    valid_c = Z_c > 0
    X_c = X_c[valid_c]
    Y_c = Y_c[valid_c]
    Z_c = Z_c[valid_c]
    pts_color = np.stack([X_c, Y_c, Z_c], axis=1)

    fx_c, fy_c = K_color["fx"], K_color["fy"]
    cx_c, cy_c = K_color["cx"], K_color["cy"]
    W_c, H_c = K_color["width"], K_color["height"]

    u_c = fx_c * X_c / Z_c + cx_c
    v_c = fy_c * Y_c / Z_c + cy_c

    u_c = np.round(u_c).astype(int)
    v_c = np.round(v_c).astype(int)

    in_bounds = (u_c >= 0) & (u_c < W_c) & (v_c >= 0) & (v_c < H_c)
    u_c = u_c[in_bounds]
    v_c = v_c[in_bounds]
    Z_c = Z_c[in_bounds]
    pts_color = pts_color[in_bounds]

    depth_color_m = np.zeros((H_c, W_c), dtype=np.float32)
    for z, uu, vv in zip(Z_c, u_c, v_c):
        if depth_color_m[vv, uu] == 0 or z < depth_color_m[vv, uu]:
            depth_color_m[vv, uu] = z

    color_rgb = cv2.cvtColor(color_img_bgr, cv2.COLOR_BGR2RGB)
    cols_color = color_rgb[v_c, u_c].astype(np.float32) / 255.0

    return depth_color_m, pts_color, cols_color


# =========================================================
# 2. 读取之前保存的标定 & 图像
# =========================================================
with open(os.path.join(out_dir, "ir_intrinsics.json"), "r") as f:
    K_ir = json.load(f)

with open(os.path.join(out_dir, "color_intrinsics.json"), "r") as f:
    K_color = json.load(f)

with open(os.path.join(out_dir, "ir2color_extrinsics.json"), "r") as f:
    extr = json.load(f)
R_ir2color = np.array(extr["R"], dtype=np.float32)
t_ir2color = np.array(extr["t"], dtype=np.float32)

color = cv2.imread(os.path.join(out_dir, "color_raw_0000.png"), cv2.IMREAD_COLOR)

# =========================================================
# 3. 读取 FoundationStereo 输出的 disparity/depth
# =========================================================
disp = np.load(fs_disp_path).astype(np.float32)    # HxW, float32 / 这里已经是深度了，而不是视差（在run.demo里转换过了）


# =========================================================
# 4. 做 IR→RGB 对齐 + 点云输出
# =========================================================
depth_color_from_fs, pts_color_fs, cols_color_fs = project_ir_depth_to_color(
    depth_ir_m=disp,
    K_ir=K_ir,
    K_color=K_color,
    R_ir2color=R_ir2color,
    t_ir2color=t_ir2color,
    color_img_bgr=color,
)

if pts_color_fs is not None:
    # 保存点云
    pcd_fs = o3d.geometry.PointCloud()
    pcd_fs.points = o3d.utility.Vector3dVector(pts_color_fs)
    pcd_fs.colors = o3d.utility.Vector3dVector(cols_color_fs)

    ply_fs_path = os.path.join(out_dir, "cloud_fs_ir2rgb.ply")
    o3d.io.write_point_cloud(ply_fs_path, pcd_fs)
    print("FS-IR depth projected to RGB frame, point cloud saved to:", ply_fs_path)

    # 保存对齐后的深度图（16-bit png）
    depth_fs_u16 = (depth_color_from_fs * 1000.0).astype(np.uint16)
    cv2.imwrite(os.path.join(out_dir, "depth_fs_ir2rgb_0000.png"), depth_fs_u16)
    print("Aligned FS depth saved to depth_fs_ir2rgb_0000.png")

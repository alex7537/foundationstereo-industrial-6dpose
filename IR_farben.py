import os
import json
import numpy as np
import cv2
import open3d as o3d

# =========================================================
# Config
# =========================================================
out_dir = "/home/match/FS/FoundationStereo/shared_fs_test"
fs_depth_path = os.path.join(out_dir, "depth_meter.npy")  # FS 输出：单位米的深度图（IR-left分辨率）
color_path = os.path.join(out_dir, "color_raw_0000.png")

ir_intr_path = os.path.join(out_dir, "ir_intrinsics.json")
color_intr_path = os.path.join(out_dir, "color_intrinsics.json")
ir2color_extr_path = os.path.join(out_dir, "ir2color_extrinsics.json")

ply_out = os.path.join(out_dir, "cloud_fs_ir2rgb_colored.ply")
depth_png_out = os.path.join(out_dir, "depth_fs_ir2rgb_0000.png")

# =========================================================
# Helpers
# =========================================================
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def assert_intr_matches_image(K: dict, img: np.ndarray, name: str):
    H, W = img.shape[:2]
    assert K["width"] == W and K["height"] == H, (
        f"[{name}] intrinsics size mismatch: "
        f"K says (W,H)=({K['width']},{K['height']}), "
        f"image is (W,H)=({W},{H})"
    )

def backproject_depth_to_points(depth_m: np.ndarray, K: dict):
    """
    depth_m: (H, W) depth in meters, in the same resolution as K.
    return:
      pts_ir: (N,3) in IR camera frame
      uv_ir:  (N,2) original IR pixel coords (u,v) (optional use)
    """
    H, W = depth_m.shape
    assert H == K["height"] and W == K["width"], "Depth size must match IR intrinsics."

    fx, fy = K["fx"], K["fy"]
    cx, cy = K["cx"], K["cy"]

    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    z = depth_m.astype(np.float32)

    valid = np.isfinite(z) & (z > 0)
    if not np.any(valid):
        return None, None

    u = u[valid].astype(np.float32)
    v = v[valid].astype(np.float32)
    z = z[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    pts = np.stack([x, y, z], axis=1)      # (N,3)
    uv = np.stack([u, v], axis=1)          # (N,2)
    return pts, uv

def transform_points(pts: np.ndarray, R: np.ndarray, t: np.ndarray):
    """
    pts: (N,3), R: (3,3), t: (3,) or (3,1)
    return: (N,3)
    """
    assert R.shape == (3,3)
    t = t.reshape(3,1)
    pts_T = pts.T  # (3,N)
    out = (R @ pts_T + t).T
    return out

def project_points_to_pixels(pts: np.ndarray, K: dict):
    """
    pts: (N,3) in that camera frame
    return: u (N,), v (N,), z (N,)
    """
    fx, fy = K["fx"], K["fy"]
    cx, cy = K["cx"], K["cy"]
    X, Y, Z = pts[:,0], pts[:,1], pts[:,2]

    valid = np.isfinite(Z) & (Z > 0)
    X, Y, Z = X[valid], Y[valid], Z[valid]

    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    return u, v, Z, valid

def zbuffer_select(u, v, z, pts, colors, H, W):
    """
    对每个像素(u,v)只保留最近深度的点，避免遮挡/穿帮上色。
    u,v: float (N,)
    z:   float (N,)
    pts: (N,3)
    colors: (N,3)
    return: filtered pts/colors + depth map (H,W)
    """
    # 量化到像素格（先round，你也可换成floor）
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & np.isfinite(z) & (z > 0)
    ui, vi, z = ui[inb], vi[inb], z[inb]
    pts = pts[inb]
    colors = colors[inb]

    if ui.size == 0:
        return None, None, None

    # 展平像素索引
    pix = vi * W + ui  # (M,)

    # 对 pix 排序，然后在每个 pix 组内取最小 z（最近）
    order = np.lexsort((z, pix))  # 先按pix，再按z
    pix_s = pix[order]
    z_s = z[order]
    pts_s = pts[order]
    col_s = colors[order]
    ui_s = ui[order]
    vi_s = vi[order]

    # 每个pix第一次出现就是最小z
    keep = np.ones_like(pix_s, dtype=bool)
    keep[1:] = pix_s[1:] != pix_s[:-1]

    pts_keep = pts_s[keep]
    col_keep = col_s[keep]
    ui_keep = ui_s[keep]
    vi_keep = vi_s[keep]
    z_keep = z_s[keep]

    depth_map = np.zeros((H, W), dtype=np.float32)
    depth_map[vi_keep, ui_keep] = z_keep

    return pts_keep, col_keep, depth_map

# =========================================================
# Main
# =========================================================
def main():
    # Load
    K_ir = load_json(ir_intr_path)
    K_color = load_json(color_intr_path)
    extr = load_json(ir2color_extr_path)
    
    R = np.array(extr["R"], dtype=np.float32)
    t = np.array(extr["t"], dtype=np.float32)
    print("t (raw):", t.reshape(-1))
    print("t norm:", np.linalg.norm(t))
    color_bgr = cv2.imread(color_path, cv2.IMREAD_COLOR)
    assert color_bgr is not None, f"Failed to read color image: {color_path}"

    depth_ir_m = np.load(fs_depth_path).astype(np.float32)
    assert depth_ir_m.ndim == 2, "FS depth must be HxW"

    # Check sizes
    assert depth_ir_m.shape == (K_ir["height"], K_ir["width"]), (
        f"FS depth shape {depth_ir_m.shape} != IR intrinsics (H,W)=({K_ir['height']},{K_ir['width']})"
    )
    assert_intr_matches_image(K_color, color_bgr, "Color")

    Hc, Wc = color_bgr.shape[:2]

    # 1) Backproject IR depth -> IR 3D points
    pts_ir, _ = backproject_depth_to_points(depth_ir_m, K_ir)
    if pts_ir is None:
        print("No valid depth points in FS depth map.")
        return

    # 2) IR frame -> Color camera frame
    pts_c = transform_points(pts_ir, R, t)

    # 3) Project to color pixels
    u, v, z, valid_mask = project_points_to_pixels(pts_c, K_color)
    pts_c_valid = pts_c[valid_mask]

    # 4) Sample color (nearest) for each projected point (before z-buffer)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    inb = (ui >= 0) & (ui < Wc) & (vi >= 0) & (vi < Hc)

    ui = ui[inb]; vi = vi[inb]
    z = z[inb]
    pts_c_valid = pts_c_valid[inb]

    cols = color_rgb[vi, ui]  # (N,3)

    if ui.size == 0:
        print("All projected points are out of RGB bounds. Check extrinsics/intrinsics.")
        return

    # 5) Z-buffer select: keep nearest per pixel + make aligned depth map
    pts_keep, col_keep, depth_color_m = zbuffer_select(
        u=ui.astype(np.float32),  # pass pixel coords already int-ish
        v=vi.astype(np.float32),
        z=z,
        pts=pts_c_valid,
        colors=cols,
        H=Hc, W=Wc
    )

    if pts_keep is None:
        print("Z-buffer selection kept no points. Check inputs.")
        return

    # 1) 深度范围裁剪（按你实验距离改）
    z = pts_keep[:, 2]
    mask = (z > 0.1) & (z < 2.0)   # 例如只保留 10cm~2m
    pts_keep = pts_keep[mask]
    col_keep = col_keep[mask]

    # 2) Open3D 统计离群点去除（可选，但很有效）
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_keep)
    pcd.colors = o3d.utility.Vector3dVector(col_keep)

    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    o3d.io.write_point_cloud(ply_out, pcd)

    print("Saved:")
    print("  Colored point cloud:", ply_out)
    print("  RGB-aligned depth (16-bit mm):", depth_png_out)
    print(f"  Points before z-buffer: {pts_c_valid.shape[0]}")
    print(f"  Points after  z-buffer: {pts_keep.shape[0]}")

if __name__ == "__main__":
    main()

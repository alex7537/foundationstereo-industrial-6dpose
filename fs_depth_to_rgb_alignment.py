#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fs_depth_to_rgb_alignment.py

对 Intel RealSense D435：
- 获取 depth / color 的内外参
- 将 FoundationStereo 的深度 (depth/IR 坐标系) 对齐到 RGB 相机坐标系
- 生成 depth_fs_aligned.png，可直接作为 RGB-D 输入给 FoundationPose

依赖:
    pip install pyrealsense2 opencv-python numpy
"""

import json
import os
from dataclasses import dataclass

import cv2
import numpy as np
import pyrealsense2 as rs


# ----------------------------------------------------------------------
# 1. 数据结构
# ----------------------------------------------------------------------

@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass
class Extrinsics:
    """depth 相机坐标系 -> color 相机坐标系 的变换"""
    R: np.ndarray  # 3x3
    t: np.ndarray  # 3,


# ----------------------------------------------------------------------
# 2. 从 RealSense 读取标定参数hon fs_depth_to_rgb_alignment.py --depth_fs ./outputs_test/fs_depth.npy

# ----------------------------------------------------------------------

def get_rs_d435_calib():
    """
    连接到 D435，相机开启后读取:
        - depth intrinsics
        - color intrinsics
        - depth -> color extrinsics
        - depth scale
    返回: (K_depth, K_color, extr_dc, depth_scale)
    """
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("No RealSense device detected")

    pipeline = rs.pipeline()
    config = rs.config()

    # 分辨率可以根据你平时用的来改，保证和 FoundationStereo 输入一致
    W, H = 640, 480
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    # 拿到 profile
    depth_profile = profile.get_stream(rs.stream.depth) \
        .as_video_stream_profile()
    color_profile = profile.get_stream(rs.stream.color) \
        .as_video_stream_profile()

    depth_intr = depth_profile.get_intrinsics()
    color_intr = color_profile.get_intrinsics()

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # depth -> color 的外参
    extr_dc_rs = depth_profile.get_extrinsics_to(color_profile)
    R = np.array(extr_dc_rs.rotation).reshape(3, 3)
    t = np.array(extr_dc_rs.translation)  # 单位: 米

    K_depth = CameraIntrinsics(
        fx=depth_intr.fx,
        fy=depth_intr.fy,
        cx=depth_intr.ppx,
        cy=depth_intr.ppy,
        width=depth_intr.width,
        height=depth_intr.height,
    )
    K_color = CameraIntrinsics(
        fx=color_intr.fx,
        fy=color_intr.fy,
        cx=color_intr.ppx,
        cy=color_intr.ppy,
        width=color_intr.width,
        height=color_intr.height,
    )
    extr_dc = Extrinsics(R=R, t=t)

    pipeline.stop()
    print("RealSense calibration loaded:")
    print("  depth  K =", K_depth)
    print("  color  K =", K_color)
    print("  depth_scale =", depth_scale)
    print("  R_dc =\n", R)
    print("  t_dc =", t)

    return K_depth, K_color, extr_dc, depth_scale


def save_calib_json(path="rs_calib_d435.json"):
    """一次性导出标定到 json，之后离线使用"""
    Kd, Kc, extr, depth_scale = get_rs_d435_calib()
    data = {
        "depth_intrinsics": Kd.__dict__,
        "color_intrinsics": Kc.__dict__,
        "extrinsics_depth_to_color": {
            "R": extr.R.tolist(),
            "t": extr.t.tolist(),
        },
        "depth_scale": depth_scale,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Calibration saved to: {path}")


def load_calib_json(path="rs_calib_d435.json"):
    """从 json 读取标定参数（不用每次都连相机）"""
    with open(path, "r") as f:
        data = json.load(f)

    kd = data["depth_intrinsics"]
    kc = data["color_intrinsics"]
    ex = data["extrinsics_depth_to_color"]

    Kd = CameraIntrinsics(**kd)
    Kc = CameraIntrinsics(**kc)
    extr = Extrinsics(
        R=np.array(ex["R"], dtype=float),
        t=np.array(ex["t"], dtype=float),
    )
    depth_scale = float(data["depth_scale"])
    return Kd, Kc, extr, depth_scale


# ----------------------------------------------------------------------
# 3. 深度对齐核心函数
# ----------------------------------------------------------------------

def depth_to_points(depth, K: CameraIntrinsics):
    """
    将 depth 图 (H, W) back-project 为 3D 点 (N, 3)，在该相机坐标系下。
    depth 单位: 米
    """
    assert depth.shape == (K.height, K.width)
    H, W = depth.shape

    vs, us = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    Z = depth  # (H, W)
    valid = Z > 0  # 有效深度 mask

    us = us[valid]
    vs = vs[valid]
    Z = Z[valid]

    X = (us - K.cx) * Z / K.fx
    Y = (vs - K.cy) * Z / K.fy

    pts = np.stack((X, Y, Z), axis=1)  # (N, 3)
    return pts, us, vs


def project_points_to_image(pts, K: CameraIntrinsics):
    """
    将 3D 点 (N,3) 投影到图像平面，返回浮点像素坐标 u,v 和深度 z_c
    """
    X = pts[:, 0]
    Y = pts[:, 1]
    Z = pts[:, 2]

    # 避免除零
    eps = 1e-6
    Z_safe = np.where(Z < eps, eps, Z)

    u = K.fx * X / Z_safe + K.cx
    v = K.fy * Y / Z_safe + K.cy
    return u, v, Z


def align_depth_fs_to_rgb(
    depth_fs,
    K_depth: CameraIntrinsics,
    K_color: CameraIntrinsics,
    extr_dc: Extrinsics,
    out_shape=None,
):
    """
    把 FoundationStereo 生成的 depth (在 depth/IR 坐标系下) 对齐到 RGB 相机图像平面。

    参数:
        depth_fs : np.ndarray, shape (H, W), float32/float64, 单位: 米
        K_depth  : depth 相机内参 (FoundationStereo 对应的相机)
        K_color  : RGB 相机内参
        extr_dc  : Extrinsics(depth->color)
        out_shape: (Hc, Wc)，输出深度图大小，默认使用 K_color 的分辨率

    返回:
        depth_rgb: np.ndarray, shape (Hc, Wc)，float32，单位: 米
    """
    if out_shape is None:
        out_H, out_W = K_color.height, K_color.width
    else:
        out_H, out_W = out_shape

    depth_fs = depth_fs.astype(np.float32)
    assert depth_fs.shape == (K_depth.height, K_depth.width)

    # 1. depth 相机坐标下的 3D 点
    pts_d, _, _ = depth_to_points(depth_fs, K_depth)  # (N,3)

    # 2. 变换到 color 相机坐标系: Pc = R * Pd + t
    R = extr_dc.R.astype(np.float32)
    t = extr_dc.t.astype(np.float32).reshape(3, 1)
    pts_d_T = pts_d.T  # 3 x N
    pts_c = (R @ pts_d_T + t).T  # (N, 3)

    # 3. 投影到 RGB 图像平面
    u, v, Zc = project_points_to_image(pts_c, K_color)

    # 4. 光栅化成深度图: 对每个像素取最近的 z
    depth_rgb = np.zeros((out_H, out_W), dtype=np.float32)
    depth_rgb[:, :] = 0.0  # 0 表示无效

    u_rounded = np.round(u).astype(int)
    v_rounded = np.round(v).astype(int)

    # 只保留在图像范围内 & 正深度
    valid = (
        (u_rounded >= 0)
        & (u_rounded < out_W)
        & (v_rounded >= 0)
        & (v_rounded < out_H)
        & (Zc > 0)
    )

    u_valid = u_rounded[valid]
    v_valid = v_rounded[valid]
    z_valid = Zc[valid]

    # 为处理“同一像素多个点”问题，先初始化为 +inf，然后取 min
    depth_tmp = np.full((out_H, out_W), np.inf, dtype=np.float32)

    for ui, vi, zi in zip(u_valid, v_valid, z_valid):
        if zi < depth_tmp[vi, ui]:
            depth_tmp[vi, ui] = zi

    depth_rgb[depth_tmp < np.inf] = depth_tmp[depth_tmp < np.inf]
    # 其余仍为 0.0

    return depth_rgb


# ----------------------------------------------------------------------
# 4. 演示：从 FS depth 得到对齐到 RGB 的 depth_fs_aligned.png
# ----------------------------------------------------------------------

def demo_align_from_files(
    depth_fs_path,
    calib_json="rs_calib_d435.json",
    out_depth_path="depth_fs_aligned.png",
):
    """
    示例:
        读取 FS 生成的深度文件 (npy 或 16bit png)，
        加载 RealSense 标定，把 depth 对齐到 RGB，
        存成 16位PNG (mm)。
    """
    # 1. 读取深度
    if depth_fs_path.endswith(".npy"):
        depth_fs = np.load(depth_fs_path).astype(np.float32)  # 米
    else:
        # 假设 PNG 保存的是 mm
        depth_raw = cv2.imread(depth_fs_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise FileNotFoundError(depth_fs_path)
        depth_fs = depth_raw.astype(np.float32) / 1000.0  # 转成米

    # 2. 读取标定
    Kd, Kc, extr, depth_scale = load_calib_json(calib_json)

    print("Depth FS shape:", depth_fs.shape)
    print("Depth K:", Kd)
    print("Color K:", Kc)

    # 3. 对齐
    depth_aligned = align_depth_fs_to_rgb(
        depth_fs,
        K_depth=Kd,
        K_color=Kc,
        extr_dc=extr,
    )

    # 4. 保存为 16bit PNG (mm)
    depth_mm = (depth_aligned * 1000.0).astype(np.uint16)
    cv2.imwrite(out_depth_path, depth_mm)
    print("Aligned FS depth saved to:", out_depth_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Align FoundationStereo depth (depth/IR frame) to RGB camera for RealSense D435"
    )
    parser.add_argument(
        "--save_calib",
        action="store_true",
        help="Only grab calibration from RealSense and save to rs_calib_d435.json",
    )
    parser.add_argument(
        "--depth_fs",
        type=str,
        default="",
        help="Path to FS depth file (.npy or 16bit .png in mm)",
    )
    parser.add_argument(
        "--calib_json",
        type=str,
        default="rs_calib_d435.json",
        help="Calibration json file path",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="depth_fs_aligned.png",
        help="Output aligned depth png (16-bit, mm)",
    )

    args = parser.parse_args()

    if args.save_calib:
        save_calib_json(args.calib_json)
    elif args.depth_fs:
        demo_align_from_files(
            depth_fs_path=args.depth_fs,
            calib_json=args.calib_json,
            out_depth_path=args.out,
        )
    else:
        print("Usage examples:")
        print("  1) 只导出标定: ")
        print("     python fs_depth_to_rgb_alignment.py --save_calib")
        print("  2) 对齐 FS 深度: ")
        print("     python fs_depth_to_rgb_alignment.py --depth_fs fs_depth.npy")

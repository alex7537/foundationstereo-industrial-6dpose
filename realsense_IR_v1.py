import pyrealsense2 as rs
import numpy as np
import cv2
import json
import os
import open3d as o3d   # ⭐ 新增：用于保存点云

out_dir = "rs_test_output"
os.makedirs(out_dir, exist_ok=True)

# 1. 配置并启动管线
pipeline = rs.pipeline()
config = rs.config()

# 分辨率可以按需改，后面最好和你算法一致
W, H = 640, 480
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
config.enable_stream(rs.stream.infrared, 1, W, H, rs.format.y8, 30)  # 左 IR
config.enable_stream(rs.stream.infrared, 2, W, H, rs.format.y8, 30)  # 右 IR

# 对齐 depth → color
align = rs.align(rs.stream.color)

print("Starting pipeline...")
profile = pipeline.start(config)

# 深度 scale（把 uint16 转成米要用）
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
print("Depth scale:", depth_scale)

# 2. 跳过前面几帧，让自动曝光稳定
for _ in range(30):
    frames = pipeline.wait_for_frames()

frames = pipeline.wait_for_frames()
aligned = align.process(frames)

color_frame = aligned.get_color_frame()
depth_frame = aligned.get_depth_frame()

ir_left_frame  = frames.get_infrared_frame(1)
ir_right_frame = frames.get_infrared_frame(2)

if not color_frame or not depth_frame:
    raise RuntimeError("No frames received")

# 3. 转成 numpy
color = np.asanyarray(color_frame.get_data())       # HxWx3, uint8 (BGR)
depth = np.asanyarray(depth_frame.get_data())       # HxW, uint16
ir_left  = np.asanyarray(ir_left_frame.get_data())  # HxW, uint8
ir_right = np.asanyarray(ir_right_frame.get_data()) # HxW, uint8

# ✅ 灰度 → 伪三通道 RGB (和你 C++ 里的 cv::cvtColor 一样)
ir_left_rgb  = cv2.cvtColor(ir_left,  cv2.COLOR_GRAY2RGB)   # HxWx3
ir_right_rgb = cv2.cvtColor(ir_right, cv2.COLOR_GRAY2RGB)   # HxWx3

print("Color shape:", color.shape)
print("Depth shape:", depth.shape)

# =============================
#  🔥 正确提取 IR stereo 内参 🔥
# =============================

intr_left = ir_left_frame.profile.as_video_stream_profile().intrinsics
intr_right = ir_right_frame.profile.as_video_stream_profile().intrinsics

# stereo baseline (单位：米)
extr = ir_left_frame.profile.get_extrinsics_to(ir_right_frame.profile)
baseline = np.linalg.norm(extr.translation)

K_left = {
    "width": intr_left.width,
    "height": intr_left.height,
    "fx": intr_left.fx,
    "fy": intr_left.fy,
    "cx": intr_left.ppx,
    "cy": intr_left.ppy,
    "baseline_m": baseline
}

print("IR Left K = ", K_left)
print("Baseline = ", baseline)

# =============================
#  ⭐ 新增：用 color + depth 直接生成 RealSense 点云 ⭐
# =============================

# 使用对齐到 color 的 depth，对应的内参也用 color 相机的
intr_color = color_frame.profile.as_video_stream_profile().intrinsics
fx, fy = intr_color.fx, intr_color.fy
cx, cy = intr_color.ppx, intr_color.ppy

# depth 转米
z = depth.astype(np.float32) * depth_scale  # HxW, float32 (meters)

H_d, W_d = z.shape
vs, us = np.meshgrid(np.arange(H_d), np.arange(W_d), indexing="ij")

X = (us - cx) * z / fx
Y = (vs - cy) * z / fy
Z = z

valid = Z > 0
pts = np.stack((X[valid], Y[valid], Z[valid]), axis=-1)

# 颜色从 BGR → RGB，再归一化
color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
cols = color_rgb[valid].reshape(-1, 3) / 255.0

pcd_rs = o3d.geometry.PointCloud()
pcd_rs.points = o3d.utility.Vector3dVector(pts)
pcd_rs.colors = o3d.utility.Vector3dVector(cols)

ply_path = os.path.join(out_dir, "cloud_rs_capture.ply")
o3d.io.write_point_cloud(ply_path, pcd_rs)
print(f"RealSense point cloud saved to: {ply_path}")
print("Valid points:", pts.shape[0])

# 5. 保存图像和 IR 内参
cv2.imwrite(os.path.join(out_dir, "color_0000.png"), color)
cv2.imwrite(os.path.join(out_dir, "depth_0000.png"), depth)           # uint16, 原始depth
cv2.imwrite(os.path.join(out_dir, "ir_left_0000.png"), ir_left_rgb)
cv2.imwrite(os.path.join(out_dir, "ir_right_0000.png"), ir_right_rgb)

# save intrinsics for stereo (IR)
with open(os.path.join(out_dir, "ir_intrinsics.json"), "w") as f:
    json.dump(K_left, f, indent=2)

pipeline.stop()
print("Saved images, intrinsics and point cloud to:", out_dir)

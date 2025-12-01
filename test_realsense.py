import pyrealsense2 as rs
import numpy as np
import cv2
import json
import os

out_dir = "rs_test_output"
os.makedirs(out_dir, exist_ok=True)

# 1. 配置并启动管线
pipeline = rs.pipeline()
config = rs.config()

# 分辨率可以按需改，后面最好和你算法一致
W, H = 640, 480
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)

# 对齐 depth → color
align = rs.align(rs.stream.color)

print("Starting pipeline...")
profile = pipeline.start(config)

# 2. 跳过前面几帧，让自动曝光稳定
for _ in range(30):
    frames = pipeline.wait_for_frames()

frames = pipeline.wait_for_frames()
aligned = align.process(frames)

color_frame = aligned.get_color_frame()
depth_frame = aligned.get_depth_frame()

if not color_frame or not depth_frame:
    raise RuntimeError("No frames received")

# 3. 转成 numpy
color = np.asanyarray(color_frame.get_data())
depth = np.asanyarray(depth_frame.get_data())  # uint16

print("Color shape:", color.shape)
print("Depth shape:", depth.shape)

# 4. 读取内参
intr = color_frame.profile.as_video_stream_profile().intrinsics
K = np.array([
    [intr.fx,      0,          intr.ppx],
    [0,            intr.fy,    intr.ppy],
    [0,            0,          1]
], dtype=float)

print("Camera intrinsics K =\n", K)

# 5. 保存结果，方便后面直接作为 demo_data 用
cv2.imwrite(os.path.join(out_dir, "color_0000.png"), color)
cv2.imwrite(os.path.join(out_dir, "depth_0000.png"), depth)

with open(os.path.join(out_dir, "intrinsics.json"), "w") as f:
    json.dump({
        "width": intr.width,
        "height": intr.height,
        "fx": intr.fx,
        "fy": intr.fy,
        "cx": intr.ppx,
        "cy": intr.ppy
    }, f, indent=2)

pipeline.stop()
print("Saved images and intrinsics to:", out_dir)

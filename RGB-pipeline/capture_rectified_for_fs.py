import pyrealsense2 as rs
import numpy as np
import cv2
import os

# ================== 用户需要改的部分 ==================
# 两个 D435 的序列号（rs-enumerate-devices 看）
LEFT_SERIAL = "727212070348"    # 左 D435
RIGHT_SERIAL = "923322072633"   # 右 D435

# 分辨率：要和标定时一致！
W, H = 1280, 720
FPS = 15

# 标定结果文件
CALIB_FILE = "stereo_calib_rgb.npz"

# FoundationStereo 仓库里的 assets 路径（自己改成你的）
FS_ASSETS_DIR = "/home/match/FS/FoundationStereo/shared_fs_test"

LEFT_OUT_PATH = os.path.join(FS_ASSETS_DIR, "left_rect.png")
RIGHT_OUT_PATH = os.path.join(FS_ASSETS_DIR, "right_rect.png")

# 也可以顺便存一份原始图像，方便调试
DEBUG_OUT_DIR = "fs_dual_rgb_debug"
os.makedirs(DEBUG_OUT_DIR, exist_ok=True)
# =====================================================


def main():
    # 1. 读取标定参数
    calib = np.load(CALIB_FILE)
    K_left = calib["K_left"]
    K_right = calib["K_right"]
    dist_left = calib["dist_left"]
    dist_right = calib["dist_right"]
    R1 = calib["R1"]
    R2 = calib["R2"]
    P1 = calib["P1"]
    P2 = calib["P2"]

    print("Loaded calibration from:", CALIB_FILE)

    # 2. 初始化两个 D435 的 RGB 管线
    pipeline_left = rs.pipeline()
    config_left = rs.config()
    config_left.enable_device(LEFT_SERIAL)
    config_left.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)

    pipeline_right = rs.pipeline()
    config_right = rs.config()
    config_right.enable_device(RIGHT_SERIAL)
    config_right.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)

    print("Starting left pipeline...")
    profile_left = pipeline_left.start(config_left)
    print("Starting right pipeline...")
    profile_right = pipeline_right.start(config_right)

    try:
        # 3. 预热几帧
        for _ in range(30):
            _ = pipeline_left.wait_for_frames()
            _ = pipeline_right.wait_for_frames()

        # 4. 获取一帧左右图
        frames_l = pipeline_left.wait_for_frames()
        frames_r = pipeline_right.wait_for_frames()

        color_frame_l = frames_l.get_color_frame()
        color_frame_r = frames_r.get_color_frame()
        if not color_frame_l or not color_frame_r:
            raise RuntimeError("未获取到左右 RGB 帧")

        color_l = np.asanyarray(color_frame_l.get_data())
        color_r = np.asanyarray(color_frame_r.get_data())

        # 保存原始图像（可选）
        cv2.imwrite(os.path.join(DEBUG_OUT_DIR, "left_raw.png"), color_l)
        cv2.imwrite(os.path.join(DEBUG_OUT_DIR, "right_raw.png"), color_r)
        print("Raw images saved to", DEBUG_OUT_DIR)

        # 5. 根据标定做 rectification
        # 注意：size 为 (W, H)，与采集图像一致
        map1_l, map2_l = cv2.initUndistortRectifyMap(
            K_left, dist_left, R1, P1, (W, H), cv2.CV_32FC1
        )
        map1_r, map2_r = cv2.initUndistortRectifyMap(
            K_right, dist_right, R2, P2, (W, H), cv2.CV_32FC1
        )

        left_rect = cv2.remap(color_l, map1_l, map2_l, cv2.INTER_LINEAR)
        right_rect = cv2.remap(color_r, map1_r, map2_r, cv2.INTER_LINEAR)

        # 6. 保存 rectified 图像到 FoundationStereo 仓库 assets
        os.makedirs(FS_ASSETS_DIR, exist_ok=True)
        cv2.imwrite(LEFT_OUT_PATH, left_rect)
        cv2.imwrite(RIGHT_OUT_PATH, right_rect)

        print("Rectified left saved to :", LEFT_OUT_PATH)
        print("Rectified right saved to:", RIGHT_OUT_PATH)

        # 可选：在 debug 目录再存一份
        cv2.imwrite(os.path.join(DEBUG_OUT_DIR, "left_rect.png"), left_rect)
        cv2.imwrite(os.path.join(DEBUG_OUT_DIR, "right_rect.png"), right_rect)

        print("Done. 现在可以在 FoundationStereo 仓库里用 run_demo.py 运行了。")

    finally:
        pipeline_left.stop()
        pipeline_right.stop()
        print("Pipelines stopped.")


if __name__ == "__main__":
    main()

# A1_save_calib_images_dual_rgb.py
import pyrealsense2 as rs
import numpy as np
import cv2
import os

# 两个 D435 的序列号（用 rs-enumerate-devices 查）
LEFT_SERIAL = "727212070348"    # 左 D435
RIGHT_SERIAL = "923322072633"   # 右 D435

W, H = 1280, 720
FPS = 15

OUT_DIR = "stereo_calib_images_rgb"
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    pipeline_left = rs.pipeline()
    config_left = rs.config()
    config_left.enable_device(LEFT_SERIAL)
    config_left.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)

    pipeline_right = rs.pipeline()
    config_right = rs.config()
    config_right.enable_device(RIGHT_SERIAL)
    config_right.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)

    profile_left = pipeline_left.start(config_left)
    profile_right = pipeline_right.start(config_right)

    print("按空格保存一对图像，按 ESC 退出。")

    idx = 0
    try:
        while True:
            frames_l = pipeline_left.wait_for_frames()
            frames_r = pipeline_right.wait_for_frames()

            color_frame_l = frames_l.get_color_frame()
            color_frame_r = frames_r.get_color_frame()
            if not color_frame_l or not color_frame_r:
                continue

            img_l = np.asanyarray(color_frame_l.get_data())
            img_r = np.asanyarray(color_frame_r.get_data())

            vis = np.hstack([img_l, img_r])
            cv2.imshow("Left | Right", vis)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break
            elif key == 32:  # 空格：保存一对
                left_path = os.path.join(OUT_DIR, f"left_{idx:03d}.png")
                right_path = os.path.join(OUT_DIR, f"right_{idx:03d}.png")
                cv2.imwrite(left_path, img_l)
                cv2.imwrite(right_path, img_r)
                print("Saved pair:", left_path, right_path)
                idx += 1

    finally:
        pipeline_left.stop()
        pipeline_right.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


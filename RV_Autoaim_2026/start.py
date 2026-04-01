import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import threading
import cv2
import rclpy
import torch
from rclpy.executors import SingleThreadedExecutor

from RV_Autoaim_2026.publisher import NodePublisher
from RV_Autoaim_2026.logger import Logger
from RV_Autoaim_2026.camera import GalaxyCamera
from RV_Autoaim_2026.autoaim import Autoaim
from RV_Autoaim_2026.detector import YoloDetector
from RV_Autoaim_2026.monitor import Monitor

def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))


def main():
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    parser = argparse.ArgumentParser(description="Start RV Autoaim Program")

    # ============================
    # 基础参数
    # ============================
    parser.add_argument('--fps', type=float, default=90, help='Camera acquisition frame rate')
    parser.add_argument('--port', type=int, default=5000, help='Web server port')

    # ============================
    # 摄像机基础参数
    # ============================
    parser.add_argument('--exposure', type=float, default=10000, help='Exposure time (us)')
    parser.add_argument('--gain', type=float, default=8, help='Camera gain')

    # ============================
    # 图像增强参数
    # ============================
    parser.add_argument('--contrast', type=float, default=2.0, help='Contrast [-50,100]')
    parser.add_argument('--gamma', type=float, default=1.0, help='Gamma [0.1,10.0]')
    parser.add_argument('--color_correction', type=int, default=1, help='Enable color correction (0 or 1)')

    # ============================
    # 白平衡
    # ============================
    parser.add_argument('--wb_r', type=float, default=1.5)
    parser.add_argument('--wb_g', type=float, default=1.0)
    parser.add_argument('--wb_b', type=float, default=1.2)

    # ============================
    # 图像增强开关
    # ============================
    parser.add_argument(
        '--image_improvement',
        action='store_true',
        help='Enable image improvement for contrast/gamma/color'
    )

    # ============================
    # 兵种和红蓝方
    # ============================
    parser.add_argument("--robot_type", default="hero")
    parser.add_argument("--robot_color", default="red")
    parser.add_argument('--yaw_bias_deg', type=float, default=-0.4, help='Static yaw zero-offset compensation in degrees')
    parser.add_argument('--pitch_bias_deg', type=float, default=0.0, help='Static pitch zero-offset compensation in degrees')

    # ============================
    # Monitor / Web 开关
    # ============================
    parser.add_argument('--disable_autoaim', action='store_true', help='Disable YOLO detection and autoaim processing')
    parser.add_argument('--detector_only', action='store_true', help='Run YOLO detection only without autoaim solving')
    parser.add_argument('--enable_monitor', action='store_true', help='Enable overlay drawing')
    parser.add_argument('--enable_web', action='store_true', help='Enable web streaming')
    parser.add_argument('--jpeg_quality', type=int, default=80, help='JPEG quality for web stream')

    args = parser.parse_args()

    logger = Logger("camera_web.log")
    logger.log("Main", "正在初始化相机...")

    # ============================
    # 初始化相机
    # ============================
    camera = GalaxyCamera(
        frame_rate=args.fps,
        logger=logger,
        robot_color=args.robot_color
    )

    camera.start()
    time.sleep(1)

    # ============================
    # 设置基础参数
    # ============================
    try:
        camera.set_exposure(args.exposure)
        camera.set_gain(args.gain)
        camera.set_frame_rate(args.fps)
        logger.log("INFO", "设置基础参数成功")
    except Exception as e:
        logger.log("Error", f"设置基础参数失败: {e}")

    # ============================
    # 设置白平衡
    # ============================
    wb_r = clamp(args.wb_r, 1.0, 7.996)
    wb_g = clamp(args.wb_g, 1.0, 7.996)
    wb_b = clamp(args.wb_b, 1.0, 7.996)
    try:
        camera.set_white_balance(wb_r, wb_g, wb_b)
    except Exception as e:
        logger.log("Error", f"设置白平衡失败: {e}")

    # ============================
    # 图像增强参数
    # ============================
    if args.image_improvement:
        logger.log("Main", "启用图像增强功能")

        contrast = clamp(args.contrast, -50, 100)
        gamma = clamp(args.gamma, 0.1, 10.0)

        try:
            camera.set_contrast(contrast)
        except Exception as e:
            logger.log("Error", f"设置对比度失败: {e}")

        try:
            camera.set_gamma(gamma)
        except Exception as e:
            logger.log("Error", f"设置Gamma失败: {e}")

        try:
            camera.set_color_correction(args.color_correction)
        except Exception as e:
            logger.log("Error", f"设置颜色校正失败: {e}")

    logger.log(
        "Main",
        f"""
参数加载完成:
FPS={args.fps}
Exposure={args.exposure}
Gain={args.gain}
Contrast={args.contrast}
Gamma={args.gamma}
WB=({wb_r},{wb_g},{wb_b})
YawBiasDeg={args.yaw_bias_deg}
PitchBiasDeg={args.pitch_bias_deg}
DisableAutoaim={args.disable_autoaim}
DetectorOnly={args.detector_only}
EnableMonitor={args.enable_monitor}
EnableWeb={args.enable_web}
WebPort={args.port}
"""
    )

    autoaim = Autoaim(
        robot_type=args.robot_type,
        robot_color=args.robot_color,
        yaw_bias_deg=args.yaw_bias_deg,
        pitch_bias_deg=args.pitch_bias_deg,
        logger=logger,
    )

    # 把 monitor/web 参数写进去
    autoaim.enable_monitor = args.enable_monitor
    autoaim.enable_web = args.enable_web
    
    autoaim.monitor = Monitor(
        enable_draw=args.enable_monitor,
        enable_web=args.enable_web,
        host='0.0.0.0',
        port=args.port,
        jpeg_quality=args.jpeg_quality
    )
    autoaim.monitor.start()

    processing_thread = None
    if args.disable_autoaim:
        autoaim.reset_control_state()
        logger.log("Main", "自瞄处理已关闭")
    else:
        detector = YoloDetector(robot_color=args.robot_color, imgsz=320)

        def processing_loop():
            while autoaim.running:
                frame = camera.get_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue

                full_start = time.perf_counter()
                detection_result = detector.detect(frame)
                if args.detector_only:
                    autoaim.reset_control_state()
                    if (autoaim.enable_monitor or autoaim.enable_web) and autoaim.monitor is not None:
                        autoaim.monitor.render(
                            frame=frame,
                            boxes=[detection_result["best_target"]] if detection_result["best_target"] is not None else None,
                            pred_point=None,
                            yaw_error=0.0,
                            pitch_error=0.0,
                            distance=0.0,
                            fire_command=0,
                            best_target=detection_result["best_target"]
                        )
                else:
                    autoaim.process_frame(frame, detection_result)
                full_time = time.perf_counter() - full_start
                autoaim.record_perf(detection_result["yolo_time_s"], full_time)

        processing_thread = threading.Thread(target=processing_loop, daemon=True)
        processing_thread.start()

    # ============================
    # 启动 ROS 2 Publisher
    # ============================
    rclpy.init()
    ros_publisher = NodePublisher(autoaim)

    def ros_spin_thread():
        executor = SingleThreadedExecutor()
        executor.add_node(ros_publisher)
        executor.spin()

    ros_thread = threading.Thread(target=ros_spin_thread, daemon=True)
    ros_thread.start()

    try:
        logger.log("Main", "系统启动完成")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.log("Main", "停止中...")

        autoaim.running = False
        if processing_thread is not None:
            processing_thread.join(timeout=1.0)
        camera.stop()

        try:
            ros_publisher.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()

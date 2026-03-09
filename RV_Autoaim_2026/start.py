# RV_Autoaim_2026/start.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import time
import threading
import rclpy
from rclpy.executors import SingleThreadedExecutor
from RV_Autoaim_2026.publisher import NodePublisher
from RV_Autoaim_2026.logger import Logger
from RV_Autoaim_2026.camera import GalaxyCamera
from RV_Autoaim_2026.autoaim import Autoaim
from RV_Autoaim_2026.moniter import WebServer

def clamp(value, min_val, max_val):
    """限制数值范围"""
    return max(min_val, min(value, max_val))


def main():
    parser = argparse.ArgumentParser(description="Start RV Autoaim Program")

    # ============================
    # 基础参数
    # ============================
    parser.add_argument('--fps', type=float, default=120, help='Camera acquisition frame rate')
    parser.add_argument('--port', type=int, default=5000, help='Web server port')

    # ============================
    # 摄像机基础参数
    # ============================
    parser.add_argument('--exposure', type=float, default=10000, help='Exposure time (us)')
    parser.add_argument('--gain', type=float, default=8, help='Camera gain')

    # ============================
    # 图像增强参数
    # ============================
    parser.add_argument('--contrast', type=int, default=1.5, help='Contrast [-50,100]')
    parser.add_argument('--gamma', type=float, default=1.0, help='Gamma [0.1,10.0]')
    parser.add_argument('--color_correction', type=int, default=1, help='Enable color correction (0 or 1)')

    # ============================
    # 白平衡
    # ============================
    parser.add_argument('--wb_r', type=float, default=1.5)
    parser.add_argument('--wb_g', type=float, default=1.0)
    parser.add_argument('--wb_b', type=float, default=1.0)

    # ============================
    # 图像增强开关
    # ============================
    parser.add_argument('--image_improvement', action='store_true',
                        help='Enable image improvement for contrast/gamma/color')
    
    # ============================
    # 兵种和红蓝方
    # ============================
    parser.add_argument("--robot_type", default="hero")
    parser.add_argument("--robot_color", default="red")
    args = parser.parse_args()

    logger = Logger("camera_web.log")
    logger.log("Main", "正在初始化相机...")

    # ============================
    # 初始化相机
    # ============================
    camera = GalaxyCamera(
        frame_rate=args.fps,
        # use_image_improvement=args.image_improvement,
        logger=logger,
        robot_color = args.robot_color
    )

    camera.start()
    time.sleep(1)  # 等待采集线程稳定

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
        """
    )
    # ============================
    # 启动 Autoaim
    # ============================
    autoaim = Autoaim(
        camera=camera,
        robot_type = args.robot_type, 
        robot_color = args.robot_color
    )
    
    # ============================
    # 启动 ROS 2 Publisher
    # ============================
    rclpy.init()
    # 将 autoaim 实例传入 Publisher
    ros_publisher = NodePublisher(autoaim)
    
    # 为了不阻塞主线程，用一个线程来跑 ROS 2 的 spin
    def ros_spin_thread():
        executor = SingleThreadedExecutor()
        executor.add_node(ros_publisher)
        executor.spin()
        
    ros_thread = threading.Thread(target=ros_spin_thread, daemon=True)
    ros_thread.start()
    # logger.log("Main", "ROS 2 Publisher 节点已启动并进入后台循环")
    
    # ============================
    # 启动 Web 服务
    # ============================
    web_server = WebServer(  
        host='0.0.0.0',
        port=args.port
    )
    
    autoaim.web_server = web_server
    
    try:
        logger.log("Main", f"Web服务启动 端口={args.port}")
        # 用后台线程启动 Web 服务
        threading.Thread(target=web_server.run, daemon=True).start()

        # 主线程可以继续做其他事情，比如监控、发送数据
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.log("Main", "停止中...")
        camera.stop()


if __name__ == "__main__":
    main()

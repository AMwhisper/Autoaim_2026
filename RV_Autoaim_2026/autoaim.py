import cv2
import time
import threading
import os
import math
import numpy as np
import rclpy
import torch
from flask import Flask
from ultralytics import YOLO

from RV_Autoaim_2026.ballistic_solver import BallisticSolver
from RV_Autoaim_2026.logger import Logger
from RV_Autoaim_2026.kalman_tracker import AngleKalman, TargetKalman
from RV_Autoaim_2026.pnp_solver import ArmorPnPSolver
from RV_Autoaim_2026.monitor import Monitor

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "weight/best_320s.engine")


class Autoaim:
    def __init__(
        self,
        camera,
        robot_type='hero',
        robot_color='blue',
        yaw_bias_deg=0.0,
        pitch_bias_deg=0.0,
    ):
        self.camera = camera
        self.app = Flask(__name__)
        self.ballistic = BallisticSolver()
        self.robot_type = robot_type
        self.last_time = time.time()
        self.robot_color = robot_color
        self.yaw_bias_deg = float(yaw_bias_deg)
        self.pitch_bias_deg = float(pitch_bias_deg)
    
        # ============================
        # 性能统计
        # ============================
        self.show_perf_log = True              # 是否打印性能信息
        self.perf_log_interval = 1.0           # 每隔多少秒打印一次
        self.perf_last_log_time = time.perf_counter()

        self.yolo_fps = 0.0
        self.full_fps = 0.0
        self.yolo_time_ms = 0.0
        self.full_time_ms = 0.0

        self.perf_frame_count = 0
        self.perf_yolo_time_sum = 0.0
        self.perf_full_time_sum = 0.0
        # ============================
        # Autoaim 参数
        # ============================
        self.fire_yaw_thresh = 3.0
        self.fire_pitch_thresh = 3.0
        self.fire_lock_frames = 5
        self.lock_count = 0
        self.fire_command = 0
        self.smooth_yaw = 0.0
        self.smooth_pitch = 0.0
        self.distance = 0.0
        self.imgsz = 320
        self.command_deadband_yaw = 0.35
        self.command_deadband_pitch = 0.35
        self.command_timeout_sec = 0.035
        self.last_output_time = 0.0
        self.output_seq = 0

        # ============================
        # Monitor / Web 开关
        # ============================
        self.enable_monitor = False   # 是否绘制调试信息
        self.enable_web = False       # 是否开启网页推流

        self.monitor = Monitor(
            enable_draw=self.enable_monitor,
            enable_web=self.enable_web,
            host="0.0.0.0",
            port=5000,
            jpeg_quality=80
        )
        self.monitor.start()

        # ============================
        # 红蓝方敌人标签设置
        # ============================
        if self.robot_color.lower() == 'red':
            self.enemy_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        elif self.robot_color.lower() == 'blue':
            self.enemy_labels = [9, 10, 11, 12, 13, 14, 15, 16, 17]
        else:
            self.enemy_labels = []

        # ============================
        # 加载 YOLO 模型
        # ============================
        self.model = YOLO(model_path, task='detect')
        self.latest_frame = None
        self.lock = threading.Lock()
        self.output_condition = threading.Condition(self.lock)
        self.detect_frame_count = 0
        self.detect_fps_last_time = time.perf_counter()

        # 预热模型
        dummy_input = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.model(dummy_input, imgsz=self.imgsz, verbose=False)

        # ============================
        # 初始化卡尔曼
        # ============================
        self.kf_yaw = AngleKalman()
        self.kf_pitch = AngleKalman()
        self.kf_target = TargetKalman()
        self.last_kf_target_time = time.perf_counter()

        # ============================
        # PnP 求解器延迟初始化
        # ============================
        self.pnp_solver = None

        # ============================
        # 启动推理线程
        # ============================
        self.running = True
        self.thread = threading.Thread(target=self.autoaim_loop, daemon=True)
        self.thread.start()

    def apply_deadband(self, value: float, threshold: float) -> float:
        return 0.0 if abs(value) < threshold else float(value)


    def update_control_state(self, yaw: float, pitch: float, distance: float, fire: int):
        with self.output_condition:
            self.smooth_yaw = self.apply_deadband(yaw, self.command_deadband_yaw)
            self.smooth_pitch = self.apply_deadband(pitch, self.command_deadband_pitch)
            self.distance = distance
            self.fire_command = int(fire)
            self.last_output_time = time.perf_counter()
            self.output_seq += 1
            self.output_condition.notify_all()

    def reset_control_state(self):
        with self.output_condition:
            self.smooth_yaw = 0.0
            self.smooth_pitch = 0.0
            self.distance = 0.0
            self.fire_command = 0
            self.lock_count = 0
            self.last_output_time = time.perf_counter()
            self.output_seq += 1
            self.output_condition.notify_all()

    def get_publish_control(self):
        now = time.perf_counter()
        with self.lock:
            yaw = float(self.smooth_yaw)
            pitch = float(self.smooth_pitch)
            fire = int(self.fire_command)
            output_seq = int(self.output_seq)
            source_timestamp = float(self.last_output_time)
            age = now - source_timestamp if source_timestamp > 0.0 else float("inf")

        if age > self.command_timeout_sec:
            return 0.0, 0.0, 0, output_seq, age, source_timestamp

        return yaw, pitch, fire, output_seq, age, source_timestamp

    def wait_for_control_update(self, last_seq: int, timeout: float = 0.1):
        with self.output_condition:
            if self.output_seq == last_seq:
                self.output_condition.wait(timeout=timeout)

            now = time.perf_counter()
            yaw = float(self.smooth_yaw)
            pitch = float(self.smooth_pitch)
            fire = int(self.fire_command)
            output_seq = int(self.output_seq)
            source_timestamp = float(self.last_output_time)
            age = now - source_timestamp if source_timestamp > 0.0 else float("inf")

        if age > self.command_timeout_sec:
            return 0.0, 0.0, 0, output_seq, age, source_timestamp

        return yaw, pitch, fire, output_seq, age, source_timestamp

    def autoaim_loop(self):
        print("等待相机首帧图像...")
        frame = None
        while frame is None and self.running:
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.1)

        if not self.running:
            return

        h, w, _ = frame.shape
        self.pnp_solver = ArmorPnPSolver(w, h)

        while self.running:
            full_start = time.perf_counter()

            frame = self.camera.get_frame()
     

            if frame is None:
                continue

            # ============================
            # YOLO 推理计时
            # ============================
            torch.cuda.synchronize()
            yolo_start = time.perf_counter()
            results = self.model(
                frame,
                imgsz=self.imgsz,
                stream=False,
                half=True,
                device=0,
                verbose=False
            )
            torch.cuda.synchronize()

            yolo_end = time.perf_counter()
            yolo_time = yolo_end - yolo_start

            # 一次性取出，避免循环里反复 .cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            best_target = None
            min_center_dist = float('inf')

            h, w, _ = frame.shape
            cx_img = w / 2
            cy_img = h / 2

            # ============================
            # 选择最近中心的敌方目标
            # ============================
            for i, det in enumerate(boxes):
                label = classes[i]
                if label not in self.enemy_labels:
                    continue

                x1, y1, x2, y2 = det
                bx = (x1 + x2) / 2
                by = (y1 + y2) / 2
                center_dist = math.hypot(bx - cx_img, by - cy_img)

                if center_dist < min_center_dist:
                    min_center_dist = center_dist
                    best_target = det

            # ============================
            # 有目标
            # ============================
            if best_target is not None:
                x1, y1, x2, y2 = best_target

                # 目标中心
                bx = (x1 + x2) / 2
                by = (y1 + y2) / 2

                # 目标中心卡尔曼预测
                now_kf = time.perf_counter()
                dt = now_kf - self.last_kf_target_time
                self.last_kf_target_time = now_kf

                pred_x, pred_y = self.kf_target.predict_update(bx, by, dt)

                # PnP
                pnp_result = self.pnp_solver.solve_from_bbox(
                    pred_x, pred_y, x1, y1, x2, y2
                )

                if pnp_result is not None:
                    distance = pnp_result["distance"]
                    yaw_cam = pnp_result["yaw_cam"]
                    pitch_cam = pnp_result["pitch_cam"]

                    # 弹道补偿
                    yaw, pitch = self.ballistic.solve(
                        yaw_cam,
                        pitch_cam,
                        distance,
                        self.robot_type
                    )

                    yaw += self.yaw_bias_deg
                    pitch += self.pitch_bias_deg

                    # yaw / pitch 卡尔曼平滑
                    smooth_yaw = self.kf_yaw.predict_update(yaw)
                    smooth_pitch = self.kf_pitch.predict_update(pitch)

                    # 控制误差
                    yaw_error = -smooth_yaw
                    pitch_error = -smooth_pitch

                    # 自动开火逻辑
                    if self.robot_type.lower() == "sentry":
                        if abs(yaw_error) < self.fire_yaw_thresh and abs(pitch_error) < self.fire_pitch_thresh:
                            self.lock_count += 1
                        else:
                            self.lock_count = 0

                        if self.lock_count > self.fire_lock_frames:
                            self.fire_command = 1
                        else:
                            self.fire_command = 0
                    else:
                        self.fire_command = 0

                    current_fire_command = self.fire_command
                    self.update_control_state(
                        yaw=smooth_yaw,
                        pitch=smooth_pitch,
                        distance=distance,
                        fire=current_fire_command
                    )

                    # 调试显示 / Web 推流
                    if self.enable_monitor or self.enable_web:
                        self.monitor.render(
                            frame=frame,
                            boxes=[best_target],
                            pred_point=(pred_x, pred_y),
                            yaw_error=yaw_error,
                            pitch_error=pitch_error,
                            distance=distance,
                            fire_command=self.fire_command,
                            best_target=best_target
                        )

                else:
                    # solvePnP 失败也当作未有效锁定
                    self.fire_command = 0
                    self.reset_control_state()
                    self.kf_yaw.reset()
                    self.kf_pitch.reset()

                    if self.enable_monitor or self.enable_web:
                        self.monitor.render(
                            frame=frame,
                            boxes=[best_target],
                            pred_point=(pred_x, pred_y),
                            yaw_error=None,
                            pitch_error=None,
                            distance=None,
                            fire_command=0,
                            best_target=best_target
                        )

            # ============================
            # 无目标
            # ============================
            else:
                self.reset_control_state()
                self.kf_yaw.reset()
                self.kf_pitch.reset()
                self.kf_target.reset()

                if self.enable_monitor or self.enable_web:
                    self.monitor.render(
                        frame=frame,
                        boxes=None,
                        pred_point=None,
                        yaw_error=0.0,
                        pitch_error=0.0,
                        distance=0.0,
                        fire_command=0,
                        best_target=None
                    )

            # ============================
            # 完整链路计时
            # ============================
            full_end = time.perf_counter()
            full_time = full_end - full_start

            self.yolo_time_ms = yolo_time * 1000.0
            self.full_time_ms = full_time * 1000.0
            self.yolo_fps = 1.0 / yolo_time if yolo_time > 1e-6 else 0.0
            self.full_fps = 1.0 / full_time if full_time > 1e-6 else 0.0

            # 累积平均
            self.perf_frame_count += 1
            self.perf_yolo_time_sum += yolo_time
            self.perf_full_time_sum += full_time

            now = time.perf_counter()
            if self.show_perf_log and (now - self.perf_last_log_time) >= self.perf_log_interval:
                avg_yolo_time = self.perf_yolo_time_sum / max(self.perf_frame_count, 1)
                avg_full_time = self.perf_full_time_sum / max(self.perf_frame_count, 1)

                avg_yolo_fps = 1.0 / avg_yolo_time if avg_yolo_time > 1e-6 else 0.0
                avg_full_fps = 1.0 / avg_full_time if avg_full_time > 1e-6 else 0.0

                print(
                    f"[PERF] "
                    f"YOLO: {self.yolo_time_ms:.2f} ms ({self.yolo_fps:.2f} FPS) | "
                    f"FULL: {self.full_time_ms:.2f} ms ({self.full_fps:.2f} FPS) | "
                    f"AVG YOLO: {avg_yolo_time * 1000.0:.2f} ms ({avg_yolo_fps:.2f} FPS) | "
                    f"AVG FULL: {avg_full_time * 1000.0:.2f} ms ({avg_full_fps:.2f} FPS)"
                )

                self.perf_frame_count = 0
                self.perf_yolo_time_sum = 0.0
                self.perf_full_time_sum = 0.0
                self.perf_last_log_time = now

# RV_Autoaim_2026/yolo_web.py
import cv2
import time
import threading
import os
import math
import numpy as np
import rclpy
from flask import Flask
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from RV_Autoaim_2026.ballistic_solver import BallisticSolver

base_dir = os.path.dirname(os.path.abspath(__file__))
# model_path = os.path.join(base_dir, "best.pt")
model_path = os.path.join(base_dir, "best.engine")
class Autoaim:
    def __init__(self, camera, robot_type = 'hero', robot_color = 'blue' ):
        self.camera = camera
        self.app = Flask(__name__)
        self.ballistic = BallisticSolver()
        self.robot_type = robot_type
        self.last_time = time.time()
        self.robot_color = robot_color
        # Autoaim参数
        self.fire_yaw_thresh = 3.0
        self.fire_pitch_thresh = 3.0
        self.fire_lock_frames = 5
        self.lock_count = 0  
        self.fire_command = 0 
        self.smooth_yaw = 0.0
        self.smooth_pitch = 0.0
        
        # 红蓝方敌人标签设置
        if self.robot_color.lower() == 'red':
            # 我们是红色，敌人是蓝色
            self.enemy_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        elif self.robot_color.lower() == 'blue':
            # 我们是蓝色，敌人是红色
            self.enemy_labels = [9, 10, 11, 12, 13, 14, 15, 16, 17]
        
        # 加载 YOLO 模型
        self.model = YOLO(model_path, task='detect')
        self.latest_frame = None
        self.lock = threading.Lock()
        
        # 预热模型（防止第一帧由于内存分配卡顿）
        dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy_input, imgsz=640, verbose=False)
        
        # -----------------------
        # 启动 YOLO 推理线程
        # -----------------------
        self.running = True
        self.thread = threading.Thread(target=self.autoaim_loop, daemon=True)
        self.thread.start()
        
        # -----------------------
        # 初始化卡尔曼滤波器
        # -----------------------
        # 循环时间
        now = time.time()
        dt = now - self.last_time
        self.last_time = now  
        
        # yaw
        self.kf_yaw = KalmanFilter(dim_x=2, dim_z=1)
        self.kf_yaw.x = np.array([0., 0.])     # [angle, velocity]
        self.kf_yaw.P = np.array([
            [1., 0.],
            [0., 1.]
        ])
        self.kf_yaw.F = np.array([
            [1., dt],
            [0., 1.]
        ])
        self.kf_yaw.H = np.array([
            [1., 0.]
        ])
        self.kf_yaw.R = np.array([[0.1]])
        self.kf_yaw.Q = np.array([
            [1e-4, 0.],
            [0., 1e-4]
        ])
        # pitch
        self.kf_pitch = KalmanFilter(dim_x=2, dim_z=1)
        self.kf_pitch.x = np.array([0., 0.])
        self.kf_pitch.P = np.array([
            [1., 0.],
            [0., 1.]
        ])
        self.kf_pitch.F = np.array([
            [1., dt],
            [0., 1.]
        ])
        self.kf_pitch.H = np.array([
            [1., 0.]
        ])
        self.kf_pitch.R = np.array([[0.1]])
        self.kf_pitch.Q = np.array([
            [1e-4, 0.],
            [0., 1e-4]
        ])
        
        # 卡尔曼预测
        self.kf_target = KalmanFilter(dim_x=4, dim_z=2)
        self.kf_target.x = np.array([0.,0.,0.,0.])
        self.kf_target.F = np.array([
            [1,0,1,0],
            [0,1,0,1],
            [0,0,1,0],
            [0,0,0,1]
        ])

        self.kf_target.H = np.array([
            [1,0,0,0],
            [0,1,0,0]
        ])

        self.kf_target.P *= 10
        self.kf_target.R *= 5
        self.kf_target.Q *= 0.01
        

    # ============================
    # YOLO + solvePnP + 卡尔曼滤波循环
    # ============================
    def autoaim_loop(self):
        
        # --- 1. 等待相机准备就绪 ---
        print("等待相机首帧图像...")
        frame = None
        while frame is None and self.running:
            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.1)
        
        if not self.running: return
        
        # 相机内参 (MER-230-168U3C)
        # fx, fy = 焦距像素, cx, cy = 图像中心
        h, w, _ = self.camera.get_frame().shape
        fov_x = 49.6  # 水平 FOV
        fov_y = 30.0  # 垂直 FOV
        fx = w / (2 * np.tan(np.radians(fov_x / 2)))
        fy = h / (2 * np.tan(np.radians(fov_y / 2)))
        cx = w / 2
        cy = h / 2
        camera_matrix = np.array([[fx, 0, cx],
                                [0, fy, cy],
                                [0,  0,  1]], dtype=np.float64)
        dist_coeffs = np.zeros(4)  # 假设无畸变

        while self.running:
            frame = self.camera.get_frame()
            if frame is None:
              
                continue

            # results = self.model(frame, imgsz=640, verbose=False)
            results = self.model(
                frame, 
                imgsz=512,      
                stream=False,    
                half=True,      
                device=0,       
                verbose=False
            )
            annotated_frame = results[0].plot()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            best_target = None
            min_center_dist = float('inf')

            h, w, _ = frame.shape
            cx_img = w / 2
            cy_img = h / 2

            for i, det in enumerate(boxes):
                label = int(results[0].boxes.cls[i].cpu().numpy())
                # print(f"Label: {label}")
                # print(f"Enemy labels: {self.enemy_labels}, Current label: {label}")
                if label not in self.enemy_labels:
                    # print(f"Label: {label}")
                    continue  

                x1, y1, x2, y2 = det
                bx = (x1 + x2) / 2
                by = (y1 + y2) / 2
                center_dist = math.hypot(bx - cx_img, by - cy_img)
                # print(f"Calculated Center Distance: {center_dist}")
                
                if center_dist < min_center_dist:
                    min_center_dist = center_dist
                    best_target = det
    
            if best_target is not None:
                # print("No enemy target detected")
                x1, y1, x2, y2 = best_target
                
                # 目标中心
                bx = (x1 + x2) / 2
                by = (y1 + y2) / 2

                # 卡尔曼预测更新
                measurement = np.array([bx, by])

                self.kf_target.predict()
                self.kf_target.update(measurement)
                
                # 预测未来位置
                pred_x = self.kf_target.x[0]
                pred_y = self.kf_target.x[1]

                # 画预测点（调试）
                cv2.circle(annotated_frame,(int(pred_x),int(pred_y)),6,(0,0,255),-1)

                # 用预测中心构造bbox
                width = x2 - x1
                height = y2 - y1

                image_points = np.array([
                    [pred_x-width/2, pred_y-height/2],
                    [pred_x+width/2, pred_y-height/2],
                    [pred_x+width/2, pred_y+height/2],
                    [pred_x-width/2, pred_y+height/2]
                ], dtype=np.float64)

                # 装甲板实际尺寸 (mm)
                aspect_ratio = (x2 - x1) / abs(y2 - y1)
                if aspect_ratio > 3:
                    armor_real_width = 235
                    armor_real_height = 60
                else:
                    armor_real_width = 140
                    armor_real_height = 60

                # 真实世界坐标 (以装甲板中心为原点)
                object_points = np.array([
                    [-armor_real_width/2,  armor_real_height/2, 0],
                    [ armor_real_width/2,  armor_real_height/2, 0],
                    [ armor_real_width/2, -armor_real_height/2, 0],
                    [-armor_real_width/2, -armor_real_height/2, 0]
                ], dtype=np.float64)

                # solvePnP
                success, rvec, tvec = cv2.solvePnP(object_points, image_points,
                                                camera_matrix, dist_coeffs)
                if success:
                    # tvec = [X, Y, Z] mm，Z方向就是相机距离
                    distance = np.linalg.norm(tvec)
                    horizontal_distance = math.sqrt(tvec[0]**2 + tvec[2]**2)

                    # yaw/pitch 角度
                    yaw_cam = math.degrees(math.atan2(tvec[0], tvec[2]))
                    pitch_cam = math.degrees(math.atan2(-tvec[1], horizontal_distance))

                    # 弹道补偿
                    yaw, pitch = self.ballistic.solve(
                        yaw_cam,
                        pitch_cam,
                        distance,
                        self.robot_type
                    )

                    # 卡尔曼滤波
                    self.kf_yaw.predict()
                    self.kf_yaw.update(yaw)
                    self.smooth_yaw = float(self.kf_yaw.x[0])

                    self.kf_pitch.predict()
                    self.kf_pitch.update(pitch)
                    self.smooth_pitch = float(self.kf_pitch.x[0])
                    
                    # 计算误差
                    yaw_error = self.smooth_yaw
                    pitch_error = self.smooth_pitch
                    
                    # 自动开火条件判断（只有Sentry兵种生效）
                    if self.robot_type.lower() == "sentry":
                        if abs(yaw_error) < self.fire_yaw_thresh and abs(pitch_error) < self.fire_pitch_thresh:
                            self.lock_count += 1
                        else:
                            self.lock_count = 0

                        # 只要连续锁定目标的帧数超过一定值，就触发开火
                        if self.lock_count > self.fire_lock_frames:
                            self.fire_command = 1  
                        else:
                            self.fire_command = 0  
                    else:
                        self.fire_command = 0  # 非sentry兵种不触发开火
                        
                    cv2.putText(annotated_frame, f"Yaw: {-self.smooth_yaw:.1f} deg", (10,60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    cv2.putText(annotated_frame, f"Pitch: {self.smooth_pitch:.1f} deg", (10,90),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    cv2.putText(annotated_frame, f"Distance: {distance:.0f} mm", (10,120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv2.putText(annotated_frame, f"Fire Command: {self.fire_command}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
        
            else:
                with self.lock:
                    # 1. 数据归零
                    self.smooth_yaw = 0.0
                    self.smooth_pitch = 0.0
                    self.distance = 0.0
                    self.fire_command = 0
                    self.lock_count = 0 # 重置锁定计数，防止闪现目标误开火
                
                # 2. 重置卡尔曼滤波器，防止下次出现目标时产生巨大的跳变预测
                self.kf_yaw.x = np.array([0., 0.])
                self.kf_pitch.x = np.array([0., 0.])
                cv2.putText(annotated_frame, f"Best target selected: {best_target}", (10,60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            self.update_web(annotated_frame)
        
    def update_web(self, frame):
        with self.lock:
            self.latest_frame = frame
                
        # 向web传递最新帧
        if hasattr(self, 'web_server') and self.web_server is not None:
            self.web_server.update_frame(frame)

    

    
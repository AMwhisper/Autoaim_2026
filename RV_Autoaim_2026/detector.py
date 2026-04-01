import math
import os
import time

import numpy as np
import torch
from ultralytics import YOLO


base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "weight/best_320s.engine")


class YoloDetector:
    def __init__(self, robot_color='blue', imgsz=320):
        self.robot_color = robot_color
        self.imgsz = imgsz

        if self.robot_color.lower() == 'red':
            self.enemy_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        elif self.robot_color.lower() == 'blue':
            self.enemy_labels = [9, 10, 11, 12, 13, 14, 15, 16, 17]
        else:
            self.enemy_labels = []

        self.model = YOLO(model_path, task='detect')
        dummy_input = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.model(dummy_input, imgsz=self.imgsz, verbose=False)

    def detect(self, frame):
        torch.cuda.synchronize()
        start = time.perf_counter()
        results = self.model(
            frame,
            imgsz=self.imgsz,
            stream=False,
            half=True,
            device=0,
            verbose=False
        )
        torch.cuda.synchronize()
        yolo_time = time.perf_counter() - start

        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        best_target = None
        min_center_dist = float('inf')

        height, width, _ = frame.shape
        cx_img = width / 2
        cy_img = height / 2

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

        return {
            "boxes": boxes,
            "classes": classes,
            "best_target": best_target,
            "yolo_time_s": yolo_time,
        }

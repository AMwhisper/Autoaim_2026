import math
import numpy as np
import cv2


class ArmorPnPSolver:
    def __init__(self, frame_width: int, frame_height: int, fov_x: float = 54.2, fov_y: float = 44.6):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fov_x = fov_x
        self.fov_y = fov_y

        fx = frame_width / (2 * np.tan(np.radians(fov_x / 2)))
        fy = frame_height / (2 * np.tan(np.radians(fov_y / 2)))
        cx = frame_width / 2
        cy = frame_height / 2

        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    @staticmethod
    def get_armor_size(x1: float, y1: float, x2: float, y2: float):
        aspect_ratio = (x2 - x1) / max(abs(y2 - y1), 1e-6)
        if aspect_ratio > 3:
            return 235.0, 60.0
        return 140.0, 60.0

    def solve_from_bbox(self, pred_x: float, pred_y: float, x1: float, y1: float, x2: float, y2: float):
        width = x2 - x1
        height = y2 - y1

        image_points = np.array([
            [pred_x - width / 2, pred_y - height / 2],
            [pred_x + width / 2, pred_y - height / 2],
            [pred_x + width / 2, pred_y + height / 2],
            [pred_x - width / 2, pred_y + height / 2]
        ], dtype=np.float64)

        armor_real_width, armor_real_height = self.get_armor_size(x1, y1, x2, y2)

        object_points = np.array([
            [-armor_real_width / 2,  armor_real_height / 2, 0],
            [ armor_real_width / 2,  armor_real_height / 2, 0],
            [ armor_real_width / 2, -armor_real_height / 2, 0],
            [-armor_real_width / 2, -armor_real_height / 2, 0]
        ], dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs
        )

        if not success:
            return None

        distance = float(np.linalg.norm(tvec))
        horizontal_distance = math.sqrt(float(tvec[0] ** 2 + tvec[2] ** 2))

        yaw_cam = math.degrees(math.atan2(float(tvec[0]), float(tvec[2])))
        pitch_cam = math.degrees(math.atan2(-float(tvec[1]), horizontal_distance))

        return {
            "success": True,
            "rvec": rvec,
            "tvec": tvec,
            "distance": distance,
            "yaw_cam": yaw_cam,
            "pitch_cam": pitch_cam,
        }
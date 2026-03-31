import math
import numpy as np
from filterpy.kalman import KalmanFilter


class AngleKalman:
    def __init__(self, dt: float = 0.01, r: float = 0.1, q: float = 1e-4):
        self.kf = KalmanFilter(dim_x=2, dim_z=1)
        self.kf.x = np.array([0.0, 0.0], dtype=np.float64)  # [angle, velocity]
        self.kf.P = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], dtype=np.float64)
        self.kf.F = np.array([
            [1.0, dt],
            [0.0, 1.0]
        ], dtype=np.float64)
        self.kf.H = np.array([
            [1.0, 0.0]
        ], dtype=np.float64)
        self.kf.R = np.array([[r]], dtype=np.float64)
        self.kf.Q = np.array([
            [q, 0.0],
            [0.0, q]
        ], dtype=np.float64)

    def predict_update(self, value: float, dt: float = None) -> float:
        if dt is not None:
            self.kf.F = np.array([
                [1.0, dt],
                [0.0, 1.0]
            ], dtype=np.float64)

        self.kf.predict()
        self.kf.update(np.array([value], dtype=np.float64))
        return float(self.kf.x[0])

    def reset(self):
        self.kf.x = np.array([0.0, 0.0], dtype=np.float64)


class TargetKalman:
    def __init__(
        self,
        motion_gate_px: float = 0.0,
        velocity_decay: float = 0.3
    ):
        self.motion_gate_px = motion_gate_px
        self.velocity_decay = velocity_decay

        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.kf.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        self.kf.P *= 10
        self.kf.R *= 5
        self.kf.Q *= 0.01

    def predict_update(self, bx: float, by: float, dt: float):
        dt = max(1e-3, min(dt, 0.1))

        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        measurement = np.array([bx, by], dtype=np.float32)

        meas_dx = float(measurement[0] - self.kf.x[0])
        meas_dy = float(measurement[1] - self.kf.x[1])
        meas_speed_px = math.hypot(meas_dx, meas_dy)

        self.kf.predict()

        if meas_speed_px > self.motion_gate_px:
            self.kf.x[2] *= self.velocity_decay
            self.kf.x[3] *= self.velocity_decay
            self.kf.update(measurement)
            pred_x = float(measurement[0])
            pred_y = float(measurement[1])
        else:
            self.kf.update(measurement)
            pred_x = float(self.kf.x[0])
            pred_y = float(self.kf.x[1])

        return pred_x, pred_y

    def reset(self):
        self.kf.x = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
import threading
import time

from RV_Autoaim_2026.ballistic_solver import BallisticSolver
from RV_Autoaim_2026.kalman_tracker import AngleKalman, TargetKalman
from RV_Autoaim_2026.pnp_solver import ArmorPnPSolver


class Autoaim:
    def __init__(self, robot_type='hero', robot_color='blue', yaw_bias_deg=0.0, pitch_bias_deg=0.0, logger=None):
        self.ballistic = BallisticSolver()
        self.robot_type = robot_type
        self.robot_color = robot_color
        self.yaw_bias_deg = float(yaw_bias_deg)
        self.pitch_bias_deg = float(pitch_bias_deg)
        self.logger = logger

        self.show_perf_log = True
        self.perf_log_interval = 1.0
        self.perf_last_log_time = time.perf_counter()

        self.yolo_fps = 0.0
        self.full_fps = 0.0
        self.yolo_time_ms = 0.0
        self.full_time_ms = 0.0
        self.perf_frame_count = 0
        self.perf_yolo_time_sum = 0.0
        self.perf_full_time_sum = 0.0

        self.fire_yaw_thresh = 3.0
        self.fire_pitch_thresh = 3.0
        self.fire_lock_frames = 5
        self.lock_count = 0
        self.fire_command = 0
        self.smooth_yaw = 0.0
        self.smooth_pitch = 0.0
        self.distance = 0.0
        self.command_deadband_yaw = 0.35
        self.command_deadband_pitch = 0.35
        self.command_timeout_sec = 0.035
        self.last_output_time = 0.0
        self.output_seq = 0
        self.running = True

        self.enable_monitor = False
        self.enable_web = False
        self.monitor = None

        self.lock = threading.Lock()
        self.output_condition = threading.Condition(self.lock)

        self.kf_yaw = AngleKalman()
        self.kf_pitch = AngleKalman()
        self.kf_target = TargetKalman()
        self.last_kf_target_time = time.perf_counter()
        self.pnp_solver = None

        self.show_yaw_debug_log = False
        self.yaw_debug_log_interval = 0.02
        self.last_yaw_debug_log_time = time.perf_counter()
        self.last_yaw_error = None
        self.yaw_jump_warn_threshold = 2.5

    def initialize(self, frame_shape):
        if self.pnp_solver is not None:
            return

        _, width, _ = frame_shape
        height = frame_shape[0]
        self.pnp_solver = ArmorPnPSolver(width, height)

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

    def process_frame(self, frame, detection_result):
        if self.pnp_solver is None:
            self.initialize(frame.shape)

        best_target = detection_result.get("best_target")
        pred_point = None
        yaw_error = None
        pitch_error = None
        distance = None

        if best_target is not None:
            x1, y1, x2, y2 = best_target
            bx = (x1 + x2) / 2
            by = (y1 + y2) / 2

            now_kf = time.perf_counter()
            dt = now_kf - self.last_kf_target_time
            self.last_kf_target_time = now_kf

            pred_x, pred_y = bx, by
            pred_point = (pred_x, pred_y)

            pnp_result = self.pnp_solver.solve_from_bbox(pred_x, pred_y, x1, y1, x2, y2)

            if pnp_result is not None:
                distance = pnp_result["distance"]
                yaw_cam = pnp_result["yaw_cam"]
                pitch_cam = pnp_result["pitch_cam"]

                yaw, pitch = self.ballistic.solve(
                    yaw_cam,
                    pitch_cam,
                    distance,
                    self.robot_type
                )

                yaw += self.yaw_bias_deg
                pitch += self.pitch_bias_deg

                raw_yaw = yaw
                smooth_yaw = yaw
                smooth_pitch = pitch # 先不滤波，直接输出

                yaw_error = -smooth_yaw
                pitch_error = -smooth_pitch
                self.log_yaw_diagnostics(
                    raw_yaw=raw_yaw,
                    smooth_yaw=smooth_yaw,
                    yaw_error=yaw_error,
                    distance_mm=distance,
                    yaw_cam=yaw_cam,
                    pred_point=pred_point,
                    best_target=best_target,
                )

                if self.robot_type.lower() == "sentry":
                    if abs(yaw_error) < self.fire_yaw_thresh and abs(pitch_error) < self.fire_pitch_thresh:
                        self.lock_count += 1
                    else:
                        self.lock_count = 0

                    self.fire_command = 1 if self.lock_count > self.fire_lock_frames else 0
                else:
                    self.fire_command = 0

                self.update_control_state(
                    yaw=smooth_yaw,
                    pitch=smooth_pitch,
                    distance=distance,
                    fire=self.fire_command
                )
            else:
                self.fire_command = 0
                self.reset_control_state()
                self.kf_yaw.reset()
                self.kf_pitch.reset()
        else:
            self.reset_control_state()
            self.kf_yaw.reset()
            self.kf_pitch.reset()
            self.kf_target.reset()

        if (self.enable_monitor or self.enable_web) and self.monitor is not None:
            self.monitor.render(
                frame=frame,
                boxes=[best_target] if best_target is not None else None,
                pred_point=pred_point,
                yaw_error=0.0 if best_target is None else yaw_error,
                pitch_error=0.0 if best_target is None else pitch_error,
                distance=0.0 if best_target is None else distance,
                fire_command=self.fire_command,
                best_target=best_target
            )

    def log_yaw_diagnostics(
        self,
        raw_yaw: float,
        smooth_yaw: float,
        yaw_error: float,
        distance_mm: float,
        yaw_cam: float,
        pred_point,
        best_target,
    ):
        if self.logger is None:
            return

        now = time.perf_counter()
        yaw_jump = None if self.last_yaw_error is None else abs(yaw_error - self.last_yaw_error)
        if pred_point is None:
            pred_point_text = "(None,None)"
        else:
            pred_point_text = f"({pred_point[0]:.1f},{pred_point[1]:.1f})"

        if best_target is None:
            bbox_text = "(None)"
        else:
            bbox_text = (
                f"({best_target[0]:.1f},{best_target[1]:.1f},"
                f"{best_target[2]:.1f},{best_target[3]:.1f})"
            )

        if yaw_jump is not None and yaw_jump >= self.yaw_jump_warn_threshold:
            self.logger.log(
                "YawWarn",
                (
                    f"jump={yaw_jump:.2f}deg raw_yaw={raw_yaw:.2f}deg "
                    f"smooth_yaw={smooth_yaw:.2f}deg yaw_error={yaw_error:.2f}deg "
                    f"yaw_cam={yaw_cam:.2f}deg distance={distance_mm / 1000.0:.2f}m "
                    f"pred={pred_point_text} bbox={bbox_text}"
                )
            )

        if self.show_yaw_debug_log and (now - self.last_yaw_debug_log_time) >= self.yaw_debug_log_interval:
            self.logger.log(
                "YawDebug",
                (
                    f"raw_yaw={raw_yaw:.2f}deg smooth_yaw={smooth_yaw:.2f}deg "
                    f"yaw_error={yaw_error:.2f}deg yaw_cam={yaw_cam:.2f}deg "
                    f"distance={distance_mm / 1000.0:.2f}m pred={pred_point_text} bbox={bbox_text}"
                )
            )
            self.last_yaw_debug_log_time = now

        self.last_yaw_error = yaw_error

    def record_perf(self, yolo_time_s: float, full_time_s: float):
        self.yolo_time_ms = yolo_time_s * 1000.0
        self.full_time_ms = full_time_s * 1000.0
        self.yolo_fps = 1.0 / yolo_time_s if yolo_time_s > 1e-6 else 0.0
        self.full_fps = 1.0 / full_time_s if full_time_s > 1e-6 else 0.0

        self.perf_frame_count += 1
        self.perf_yolo_time_sum += yolo_time_s
        self.perf_full_time_sum += full_time_s

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

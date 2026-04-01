import math


class BallisticSolver:

    def __init__(self):

        # 重力
        self.g = 9.81

        # 弹速
        self.hero_speed = 12.0
        self.infantry_speed = 22.0

        # 相机 → 枪口结构 (m)
        self.hero_cam_to_muzzle_x = 0.085
        self.hero_cam_to_muzzle_z = 0.00
        self.hero_cam_to_muzzle_y = 0.13

        self.sentry_cam_to_muzzle_x = 0.0
        self.sentry_cam_to_muzzle_z = 0.00
        self.sentry_cam_to_muzzle_y = 0.09

        self.infantry_cam_to_muzzle_x = 0.0
        self.infantry_cam_to_muzzle_z = 0.00
        self.infantry_cam_to_muzzle_y = 0.09
        
    def solve(self, yaw_cam_deg, pitch_cam_deg, distance_mm, robot="hero"):

        # 单位转换
        yaw_cam = math.radians(yaw_cam_deg)
        pitch_cam = math.radians(pitch_cam_deg)
        distance = distance_mm / 1000.0

        # 选择兵种
        if robot == "hero":
            v = self.hero_speed
            cam_to_muzzle_x = self.hero_cam_to_muzzle_x
            cam_to_muzzle_y = self.hero_cam_to_muzzle_y
            cam_to_muzzle_z = self.hero_cam_to_muzzle_z
        elif robot == "sentry":
            v = self.hero_speed
            cam_to_muzzle_x = self.sentry_cam_to_muzzle_x
            cam_to_muzzle_y = self.sentry_cam_to_muzzle_y
            cam_to_muzzle_z = self.sentry_cam_to_muzzle_z
        elif robot == "infantry":
            v = self.infantry_speed
            cam_to_muzzle_x = self.infantry_cam_to_muzzle_x
            cam_to_muzzle_y = self.infantry_cam_to_muzzle_y
            cam_to_muzzle_z = self.infantry_cam_to_muzzle_z
        # 相机极坐标 → 相机坐标
        x_cam = distance * math.cos(pitch_cam) * math.sin(yaw_cam)
        y_cam = distance * math.sin(pitch_cam)
        z_cam = distance * math.cos(pitch_cam) * math.cos(yaw_cam)

        # 相机 → 枪口坐标
        x = x_cam - cam_to_muzzle_x
        y = y_cam + cam_to_muzzle_y
        z = z_cam - cam_to_muzzle_z

        d = math.sqrt(x*x + z*z)

        yaw = math.atan2(x, z)

        pitch = self.ballistic_pitch(d, y, v)

        if pitch is None:
            return yaw_cam_deg, pitch_cam_deg

        return math.degrees(yaw), math.degrees(pitch)


    def ballistic_pitch(self, distance, height, v):

        g = self.g
        v2 = v * v

        inside = v2*v2 - g*(g*distance*distance + 2*height*v2)

        if inside <= 0:
            return None

        pitch = math.atan(
            (v2 - math.sqrt(inside)) / (g*distance)
        )

        return pitch
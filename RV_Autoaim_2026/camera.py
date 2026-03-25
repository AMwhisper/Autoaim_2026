import gxipy as gx
import threading
import time
import sys
import cv2
from PIL import Image
import numpy as np

class GalaxyCamera:

    def __init__(self, frame_rate = 120, logger = None, robot_color = 'blue'):
        self.device_manager = gx.DeviceManager()
        self.cam = None
        self.latest_frame = None
        self.running = False
        self.frame_rate = frame_rate
        self.lock = threading.Lock()
        self.logger = logger
  
        self.start_time = None
        self.frames_captured = 0
        self.last_fps_time = 0
        
        self.robot_color = robot_color
        self.target_color = "red"
        
        self.open_camera()

    # ==================================================
    # 相机初始化
    # ==================================================
    def open_camera(self):
        dev_num, dev_info_list = self.device_manager.update_device_list()
        if dev_num == 0:
            print("未找到相机")
            sys.exit(1)

        strSN = dev_info_list[0].get("sn")
        self.cam = self.device_manager.open_device_by_sn(strSN)

        # 关闭触发模式
        if self.cam.TriggerMode.is_implemented():
            self.cam.TriggerMode.set(gx.GxSwitchEntry.OFF)

        # 设置帧率
        if self.cam.AcquisitionFrameRate.is_implemented():
            self.cam.AcquisitionFrameRate.set(self.frame_rate)

        # 关闭自动曝光 & 自动增益
        if self.cam.ExposureAuto.is_implemented():
            self.cam.ExposureAuto.set(gx.GxAutoEntry.OFF)
        if self.cam.GainAuto.is_implemented():
            self.cam.GainAuto.set(gx.GxAutoEntry.OFF)

        # 关闭自动白平衡
        if self.cam.BalanceWhiteAuto.is_implemented():
            self.cam.BalanceWhiteAuto.set(gx.GxAutoEntry.OFF)

        self.cam.stream_on()
        print("相机已打开")
        print("PixelFormat:", self.cam.PixelFormat.get())

    # ==================================================
    # 启动采集线程
    # ==================================================
    def start(self):
        self.running = True
        self.start_time = time.time()
        self.last_fps_time = self.start_time
        self.thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.thread.start()

    # ==================================================
    # 停止
    # ==================================================
    def stop(self):
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join()
        if self.cam:
            self.cam.stream_off()
            self.cam.close_device()
            print("相机已关闭")

    # ==================================================
    # 采集循环
    # ==================================================
    def capture_loop(self):
        while self.running:
            raw = self.cam.data_stream[0].get_image()
            if raw is None or raw.get_status() != gx.GxFrameStatusList.SUCCESS:
                continue

            # RAW Bayer → RGB
            # rotated_image = raw.raw8_rotate_90_ccw() #反转90度
            # rgb_image = rotated_image.convert("RGB")
            rgb_image = raw.convert("RGB") #不反转90度
            frame = rgb_image.get_numpy_array()

            if frame is None:
                continue

            # 线程安全缓存
            with self.lock:
                 self.latest_frame = frame # 原图

            # FPS统计
            self.frames_captured += 1
            now = time.time()
            if now - self.last_fps_time >= 1.0:
                # print(f"Camera FPS: {self.frames_captured}")
                self.frames_captured = 0
                self.last_fps_time = now

    # ==================================================
    # 获取最新帧
    # ==================================================
    def get_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            frame = self.latest_frame
            # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return frame

    # ==================================================
    # 参数控制接口
    # ==================================================
    def set_exposure(self, value):
        self.cam.ExposureTime.set(float(value))

    def get_exposure(self):
        return self.cam.ExposureTime.get()

    def set_gain(self, value):
        self.cam.Gain.set(float(value))

    def get_gain(self):
        return self.cam.Gain.get()

    def set_frame_rate(self, value):
        self.frame_rate = float(value)
        self.cam.AcquisitionFrameRate.set(self.frame_rate)

    def get_frame_rate(self):
        return self.cam.AcquisitionFrameRate.get()

    def set_white_balance(self, r, g, b):
        if not self.cam.BalanceRatioSelector.is_implemented():
            return

        self.cam.BalanceRatioSelector.set(gx.GxBalanceRatioSelectorEntry.RED)
        self.cam.BalanceRatio.set(float(r))
        self.cam.BalanceRatioSelector.set(gx.GxBalanceRatioSelectorEntry.GREEN)
        self.cam.BalanceRatio.set(float(g))
        self.cam.BalanceRatioSelector.set(gx.GxBalanceRatioSelectorEntry.BLUE)
        self.cam.BalanceRatio.set(float(b))

    def get_white_balance(self):
        self.cam.BalanceRatioSelector.set(gx.GxBalanceRatioSelectorEntry.RED)
        r = self.cam.BalanceRatio.get()
        self.cam.BalanceRatioSelector.set(gx.GxBalanceRatioSelectorEntry.GREEN)
        g = self.cam.BalanceRatio.get()
        self.cam.BalanceRatioSelector.set(gx.GxBalanceRatioSelectorEntry.BLUE)
        b = self.cam.BalanceRatio.get()
        return r, g, b
    def set_target_color(self):
        if self.robot_color == 'blue':
            self.target_color = 'red'
        elif self.robot_color == 'red':
            self.target_color = 'blue'
            
                
            
import cv2
import time
import threading
from flask import Flask, Response, render_template_string

class WebServer:
    def __init__(self, host="0.0.0.0", port=5000):
        self.host = host
        self.port = port
        self.app = Flask(__name__)

        self.latest_frame = None
        self.lock = threading.Lock()

        # 设置 Flask 路由
        self._setup_routes()
        
    # ============================
    # Flask 路由
    # ============================
    def _setup_routes(self):
        @self.app.route('/')
        def index():
            html = """
            <!doctype html>
            <html>
            <head>
                <title>RoboMaster Detection</title>
            </head>
            <body>
                <h2>Live Detection</h2>
                <img src="/video_feed" width="1280" height="1024">
            </body>
            </html>
            """
            return render_template_string(html)

        @self.app.route('/video_feed')
        def video_feed():
            return Response(
                self.gen_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )

    # ============================
    # MJPEG 推流
    # ============================
    def gen_frames(self):
        while True:
            with self.lock:
                if self.latest_frame is None:
                    time.sleep(0.01)
                    continue
                frame = self.latest_frame.copy()

            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   jpeg.tobytes() +
                   b'\r\n')
    # ============================
    # 更新最新帧
    # ============================
    def update_frame(self, frame):
        with self.lock:
            self.latest_frame = frame
            
    # ============================
    # 启动 Flask
    # ============================
    def run(self):
        self.app.run(host=self.host, port=self.port, threaded=True, debug=False)
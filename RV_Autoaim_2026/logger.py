# logger.py
import threading
import time

class Logger:
  
    def __init__(self, filename):
        self.filename = filename
        self.lock = threading.Lock()
        with open(self.filename, 'w') as f:
            f.write(f"Log started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    def log(self, tag, message):
       
        timestamp = time.strftime('%H:%M:%S')
        line = f"[{timestamp}][{tag}] {message}\n"
        with self.lock:
            
            with open(self.filename, 'a') as f:
                f.write(line)
            
            print(line, end='')

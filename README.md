# Galaxy_Camera_Web
在Nvidia Jetson Orin Nano上，用Daheng Galaxy系列相机拍摄视频并放在web端
## 1.下载并安装驱动
[Daheng官网下载中心](https://www.daheng-imaging.com/index.php?m=content&c=index&a=lists&catid=59&czxt=30&sylx=&syxj=#mmd)下载 **Galaxy Linux-armhf-Gige-U3 SDK_CN-EN** 和 **Galaxy Linux-Python SDK_CN-EN** 。  
然后先安装Gige-U3的SDK，再安装Python的SDK，不然会报错缺失libgxiapi.so。  
## 2.使用该包
在根目录（setup.py在的目录）下执行
```
pip install .
```
等待安装好后执行
```
autoaim_start
```
然后就可以通过终端输出的网址端口查看视频流了     
若别的程序只想启动相机采集的话，在python脚本中添加：
```
from galaxy_camera import GalaxyCamera
```

## 3.文件解析
camera.py是相机采集模块，web.py是网页推流服务模块，logger.py是日志输出模块，start.py是启动模块    
特别强调：采集帧率和推流帧率都是可以调的，但是推流帧率不应大于采集帧率，不然web会卡死。如果画面卡死，可以看camera和web是否有日志输出，然后进一步排查问题

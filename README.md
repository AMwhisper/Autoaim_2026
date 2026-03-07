# Galaxy_Camera_Web
在Nvidia Jetson Orin Nano上，用Daheng Galaxy系列相机采集，使用yolov8进行推理并计算Yaw和Pitch角度，并可以在Web端监控
## 1.下载并安装驱动
[Daheng官网下载中心](https://www.daheng-imaging.com/index.php?m=content&c=index&a=lists&catid=59&czxt=30&sylx=&syxj=#mmd)下载 **Galaxy Linux-armhf-Gige-U3 SDK_CN-EN** 和 **Galaxy Linux-Python SDK_CN-EN** 。  
然后先安装Gige-U3的SDK，再安装Python的SDK，不然会报错缺失libgxiapi.so。（此链接只是用于ARM架构的处理器，x86架构请自行选择下载版本） 
## 2.使用该包
在根目录（setup.py在的目录）下执行
```
pip install -e .
```
等待安装好后执行
```
autoaim_start
```
可以在后面添加参数来调整相机参数以及机器人类型和红蓝方
## 3.逻辑推理  
相机采集后，通过yolo模型推理后，取两个灯条形成的矩形范围为box，然后通过PnP算法计算出3D空间模型，并利用卡尔曼滤波器进行帧与帧之间的平滑处理，配合摄像头与枪口和轴心之间的距离等参数，计算出Yaw、Pitch轴需转动角度  
识别后将box和部分文本信息绘制在最新帧上后，推流至web端，可以通过本地和局域网查看  
## 4.注意事项  
采集帧率和推流帧率都是可以调的，但是推流帧率不应大于采集帧率，不然web会卡死。如果画面卡死，可以看日志输出，然后进一步排查问题

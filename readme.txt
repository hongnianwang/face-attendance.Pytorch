F:\Anaconda\python.exe D:/repositories/Face-Attendance/tree.py

----Face-Attendance\
    |----app.py										主要运行的文件
    |----config.py									主要是模型的配置（还包括了路径的设置，管理员密码一类的）
    |----data\										存放数据的文件夹
    |    |----facebank									人脸图像
    |    |----facebank.pth								人脸特征（模型提取后的特征）
    |    |----names.npy									人脸特征对于的标签（学号，学院，名字，性别）
    |    |----time.npy									签到的时间
    |----model\										识别模型
    |    |----model.py									识别模型
    |    |----model_cpu_final.pth						识别模型的权重
    |----mtcnn										检测模型
    |    |----mtcnn.py									检测模型
    |    |----src										一些辅助函数和权重
    |----static										一些UI的图
    |----utils.py									一些辅助函数


Process finished with exit code 0

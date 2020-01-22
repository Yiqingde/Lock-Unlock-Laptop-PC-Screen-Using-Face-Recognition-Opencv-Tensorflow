# 基于opencv tenserflow2.0实战CNN人脸识别锁定与解锁win10屏幕
# 前言
windows hello的低阶板本，没有Windows hello的3D景深镜头，因此这是一个基于图片的识别机主的程序。
具体运行时，解锁时，判断是否是本人；若不是本人或无人（10s），锁屏；若是本人，正常使用；(采取无密码原始界面)

人脸的检测采取opencv cv2.CascadeClassifier

关于模型则采取
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 62, 62, 128)       3584
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 60, 60, 64)        73792
_________________________________________________________________
flatten (Flatten)            (None, 230400)            0
_________________________________________________________________
dense (Dense)                (None, 40)                9216040
=================================================================
Total params: 9,293,416
Trainable params: 9,293,416
Non-trainable params: 0
_________________________________________________________________
None
```

基础需要由四部分组成。

face_1.py|face_2.py|face_3.py|face_4.py
-|-|-|-
制作自己人脸训练数据|由face_1.py 和 face_2.py制作的数据来进行CNN深度学习，并保存模型|由已知其他人脸来制作数据|最后的检测程序

## 运行python环境
主要是在tensorflow2.0-gpu下运行；
这里略微吐槽下tensorflow2.0.keras模块部分无提示，对于新手不太友好。
conda list：

 Name|Version|Build  Channel
-|-|-
_tflow_select|2.1.0| gpu
absl-py|      0.8.1|py37_0
altgraph|     0.17| pypi_0    pypi
astor| 0.8.0|py37_0
astroid|      2.3.3|py37_0
backcall|     0.1.0|py37_0
blas|1.0|   mkl
ca-certificates          |2019.11.27|0
certifi|      2019.11.28|  py37_0
colorama|     0.4.3|py_0
cudatoolkit|  10.0.130|0
cudnn| 7.6.5|   cuda10.0_0
cycler|0.10.0|      pypi_0    pypi
decorator|    4.4.1|py_0
future|0.18.2|      pypi_0    pypi
gast|0.2.2|py37_0
google-pasta| 0.1.8|py_0
grpcio|1.16.1       |    py37h351948d_1
h5py|2.9.0|py37h5e291fa_0
hdf5|1.10.4|  h7ebc959_0
icc_rt|2019.0.0|h0cc432a_1
intel-openmp| 2019.4|245
ipykernel|    5.1.3|py37h39e3cac_1
ipython|      7.11.1      |     py37h39e3cac_0
ipython_genutils  |        0.2.0|py37_0
isort| 4.3.21|      py37_0
jedi|0.15.2|      py37_0
joblib|0.14.1| py_0
jupyter_client|5.3.4|py37_0
jupyter_core| 4.6.1|py37_0
keras| 2.3.1|pypi_0    pypi
keras-applications       | 1.0.8|py_0
keras-preprocessing       |1.1.0|py_1
kiwisolver|   1.1.0|pypi_0    pypi
lazy-object-proxy   |      1.4.3|py37he774522_0
libprotobuf|  3.11.2|  h7bd577a_0
libsodium|    1.0.16|  h9d3ae62_0
markdown|     3.1.1|py37_0
matplotlib|   3.1.2|pypi_0    pypi
mccabe|0.6.1|py37_1
mkl| 2019.4|245
mkl-service|  2.3.0|py37hb782905_0
mkl_fft|      1.0.15    |      py37h14836fe_0
mkl_random|   1.1.0|py37h675688f_0
mouseinfo|    0.1.2|pypi_0    pypi
numpy| 1.17.4         |  py37h4320e6b_0
numpy-base|   1.17.4   |        py37hc3f5095_0
opencv-python|4.1.2.30|    pypi_0    pypi
openssl|      1.1.1d|  he774522_3
opt_einsum|   3.1.0|py_0
pandas|0.25.3|      pypi_0   | pypi
parso| 0.5.2|py_0
pefile|2019.4.18|   pypi_0   | pypi
pickleshare|  0.7.5|py37_0
pillow|7.0.0|pypi_0    |pypi
pip| 19.3.1|      py37_0
prompt_toolkit|3.0.2|py_0
protobuf|     3.11.2|           py37h33f27b4_0
pyautogui|    0.9.48|      pypi_0    pypi
pygetwindow|  0.0.8|pypi_0    pypi
pygments|     2.5.2|py_0
pyinstaller|  3.6|pypi_0    pypi
pylint|2.4.4|py37_0
pymsgbox|     1.0.7|pypi_0    pypi
pyparsing|    2.4.6|pypi_0    pypi
pyperclip|    1.7.0|pypi_0    pypi
pyreadline|   2.1|py37_1
pyrect|0.1.4|pypi_0    pypi
pyscreeze|    0.1.26|      pypi_0    pypi
python|3.7.6|   h60c2a47_2
python-dateutil   |        2.8.1|py_0
pytweening|   1.0.3|pypi_0    pypi
pytz|2019.3|      pypi_0    pypi
pywin32|      227| py37he774522_1
pywin32-ctypes|0.2.0|pypi_0    pypi
pyyaml|5.3|pypi_0    pypi
pyzmq| 18.1.0          | py37ha925a31_0
scikit-learn| 0.22.1    |       py37h6288b17_0
scipy| 1.3.2|py37h29ff71c_0
setuptools|   44.0.0|      py37_0
six| 1.13.0|      py37_0
sqlite|3.30.1|  he774522_0
tensorboard|  2.0.0| pyhb38c66f_1
tensorflow|   2.0.0          | gpu_py37h57d29ca_0
tensorflow-base        |   2.0.0 |          gpu_py37h390e234_0
tensorflow-estimator    |  2.0.0| pyh2649769_0
tensorflow-gpu|2.0.0|   h0d30ee6_0
termcolor|    1.1.0|py37_1
tornado|      6.0.3|py37he774522_0
traitlets|    4.3.3|py37_0
vc|  14.1|    h0510ff6_4
vs2015_runtime|14.16.27012   |       hf0eaf9b_1
wcwidth|      0.1.7|py37_0
werkzeug|     0.16.0| py_0
wheel| 0.33.6|      py37_0
wincertstore| 0.2|py37_0
wrapt| 1.11.2           py37he774522_0
zeromq|4.3.1|   h33f27b4_3
zlib|1.2.11|  h62dcd97_3

## 首先制作自己训练数据：
人脸数据存储至my_faces 可自己命名


face_1.py
```python

# 制作自己人脸数据

from cv2 import cv2
import os
import sys
import random

out_dir = './my_faces'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


# 改变亮度与对比度
def relight(img, alpha=1, bias=0):
    w = img.shape[1]
    h = img.shape[0]
    #image = []
    for i in range(0,w):
        for j in range(0,h):
            for c in range(3):
                tmp = int(img[j,i,c]*alpha + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j,i,c] = tmp
    return img


# 获取分类器
haar = cv2.CascadeClassifier(r'E:\ProgramData\Anaconda3\envs\tenserflow02\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

# 打开摄像头 参数为输入流，可以为摄像头或视频文件
camera = cv2.VideoCapture(0)

n = 1
while 1:
    if (n <= 5000):
        print('It`s processing %s image.' % n)
        # 读帧
        success, img = camera.read()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(gray_img, 1.3, 5)
        for f_x, f_y, f_w, f_h in faces:
            face = img[f_y:f_y+f_h, f_x:f_x+f_w]
            face = cv2.resize(face, (64,64))
            
            face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
            cv2.imshow('img', face)
            cv2.imwrite(out_dir+'/'+str(n)+'.jpg', face)
            n+=1
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break
    else:
        break


```
## 制作他人训练数据：
需要收集一个其他人脸的图片集，只要不是自己的人脸都可以，可以在网上找到，这里我给出一个我用到的图片集：
网站地址:http://vis-www.cs.umass.edu/lfw/
图片集下载:http://vis-www.cs.umass.edu/lfw/lfw.tgz
先将下载的图片集，解压到项目目录下的lfw目录下，也可以自己指定目录(修改代码中的input_dir变量)

face_3.py
```python
# -*- codeing: utf-8 -*-
import sys
import os
from cv2 import cv2

input_dir = './lfw'
output_dir = './other_faces'
size = 64

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def close_cv2():
    """删除cv窗口"""
    while(1):
        if(cv2.waitKey(100)==27):
            break
    cv2.destroyAllWindows()
# 获取分类器
haar = cv2.CascadeClassifier(r'E:\ProgramData\Anaconda3\envs\tenserflow02\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

index = 1
for (path, dirnames, filenames) in os.walk(input_dir):
    for filename in filenames:
        if filename.endswith('.jpg'):
            print('Being processed picture %s' % index)
            img_path = path+'/'+filename
            # # 从文件读取图片
            print(img_path)
            img = cv2.imread(img_path)
            # cv2.imshow(" ",img)
            # close_cv2()
            # 转为灰度图片

            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = haar.detectMultiScale(gray_img, 1.3, 5)
            for f_x, f_y, f_w, f_h in faces:
                face = img[f_y:f_y+f_h, f_x:f_x+f_w]
                face = cv2.resize(face, (64,64))
  
                # face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
                cv2.imshow('img', face)
                cv2.imwrite(output_dir+'/'+str(index)+'.jpg', face)
                index+=1
            key = cv2.waitKey(30) & 0xff
            if key == 27:
                sys.exit(0)
```
# 接下来进行数据训练
读取上文的 my_faces和other_faces文件夹下的训练数据进行训练

face_2.py
```python
# -*- codeing: utf-8 -*-
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from cv2 import cv2
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# from keras import backend as K

def getPaddingSize(img):
    h, w, _ = img.shape
    top, bottom, left, right = (0,0,0,0)
    longest = max(h, w)

    if w < longest:
        tmp = longest - w
        # //表示整除符号
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right

def readData(path, h,w,imgs,labs):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename

            img = cv2.imread(filename)
            # cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            top,bottom,left,right = getPaddingSize(img)
            # 将图片放大， 扩充图片边缘部分
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
            img = cv2.resize(img, (h, w))

            imgs.append(img)
            labs.append(path)
    return imgs,labs




def get_model():
    model = tf.keras.Sequential()
    # 第一层卷积，卷积的数量为128，卷积的高和宽是3x3，激活函数使用relu
    model.add(tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu', input_shape=(64, 64, 3)))
    # 第二层卷积
    model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
    #把多维数组压缩成一维，里面的操作可以简单理解为reshape，方便后面Dense使用
    model.add(tf.keras.layers.Flatten())
    #对应cnn的全链接层，可以简单理解为把上面的小图汇集起来，进行分类
    model.add(tf.keras.layers.Dense(40, activation='softmax'))
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return model 

def facemain():
    my_faces_path = './my_faces'
    other_faces_path = './other_faces'
    size = 64

    imgs = []
    labs = []
    imgs,labs=readData(my_faces_path,size,size,imgs,labs)
    imgs,labs=readData(other_faces_path,size,size,imgs,labs)


    # 将图片数据与标签转换成数组
    imgs = np.array(imgs)
    # labs = np.array([[0,1] if lab == my_faces_path else [1,0] for lab in labs])
    labs = np.array([[1] if lab == my_faces_path else [0] for lab in labs])
    print(imgs.shape)
    print(labs.shape)
    # 随机划分测试集与训练集
    train_x,test_x,train_y,test_y = train_test_split(imgs, labs, test_size=0.8, random_state=random.randint(0,100))

    # 参数：图片数据的总数，图片的高、宽、通道
    train_x = train_x.reshape(train_x.shape[0], size, size, 3)
    test_x = test_x.reshape(test_x.shape[0], size, size, 3)

    # 将数据转换成小于1的数
    train_x = train_x.astype('float32')/255.0
    test_x = test_x.astype('float32')/255.0

    print('train size:%s, test size:%s' % (len(train_x), len(test_x)))
    # 图片块，每次取100张图片
    batch_size = 100
    num_batch = len(train_x) // batch_size


    model=get_model()
    model.fit(train_x, train_y, epochs=5)
    model.save(r'C:\Users\Administrator\Desktop\my_model.h5')


facemain()

```

# 最后进行预测判断是否是本人,以进行是否锁屏操作
face_4.py
```python
#识别自己
from __future__ import absolute_import, division, print_function
import tensorflow as tf

from cv2 import cv2
import os
import sys
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from ctypes import *
import time
import sys


def getPaddingSize(img):
    h, w, _ = img.shape
    top, bottom, left, right = (0,0,0,0)
    longest = max(h, w)

    if w < longest:
        tmp = longest - w
        # //表示整除符号
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right

def readData(path, h,w,imgs,labs):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename

            img = cv2.imread(filename)
            # cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            top,bottom,left,right = getPaddingSize(img)
            # 将图片放大， 扩充图片边缘部分
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
            img = cv2.resize(img, (h, w))

            imgs.append(img)
            labs.append(path)
    return imgs,labs
# 改变亮度与对比度
def relight(img, alpha=1, bias=0):
    w = img.shape[1]
    h = img.shape[0]
    #image = []
    for i in range(0,w):
        for j in range(0,h):
            for c in range(3):
                tmp = int(img[j,i,c]*alpha + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j,i,c] = tmp
    return img

out_dir = './temp_faces'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# 获取分类器
haar = cv2.CascadeClassifier(r'E:\ProgramData\Anaconda3\envs\tenserflow02\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

# 打开摄像头 参数为输入流，可以为摄像头或视频文件
camera = cv2.VideoCapture(0)
n = 1

start = time.clock()
while 1:
    if (n <= 20):
        print('It`s processing %s image.' % n)
        # 读帧
        success, img = camera.read()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(gray_img, 1.3, 5)
        for f_x, f_y, f_w, f_h in faces:
            face = img[f_y:f_y+f_h, f_x:f_x+f_w]
            face = cv2.resize(face, (64,64))
            # face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
            cv2.imshow('img', face)
            cv2.imwrite(out_dir+'/'+str(n)+'.jpg', face)
            n+=1
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break
        end = time.clock()
        print(str(end-start))
        if (end-start)>10:
            user32 = windll.LoadLibrary('user32.dll')
            user32.LockWorkStation()
            sys.exit()
    else:
        break


my_faces_path = out_dir
size = 64

imgs = []
labs = []
imgs,labs=readData(my_faces_path,size,size,imgs,labs)
# 将图片数据与标签转换成数组
imgs = np.array(imgs)
# labs = np.array([[0,1] if lab == my_faces_path else [1,0] for lab in labs])
labs = np.array([[1] if lab == my_faces_path else [0] for lab in labs])
# 随机划分测试集与训练集
train_x,test_x,train_y,test_y = train_test_split(imgs, labs, test_size=0.9, random_state=random.randint(0,100))

# 参数：图片数据的总数，图片的高、宽、通道
train_x = train_x.reshape(train_x.shape[0], size, size, 3)
test_x = test_x.reshape(test_x.shape[0], size, size, 3)

# 将数据转换成小于1的数
train_x = train_x.astype('float32')/255.0
test_x = test_x.astype('float32')/255.0

restored_model = tf.keras.models.load_model(r'C:\Users\Administrator\Desktop\my_model.h5')
pre_result=restored_model.predict_classes(test_x)
print(pre_result.shape)
print(pre_result)
acc=sum(pre_result==1)/pre_result.shape[0]
print("相似度： "+str(acc))

if acc > 0.8:
    print("你是***")
else:
    user32 = windll.LoadLibrary('user32.dll')
    user32.LockWorkStation()
```

## 最后一步，添加face_4.py解锁windows运行任务计划程序库
### myface.bat 文件 
激活Anaconda环境 
切CD至face_4.py的位置
```
call activate tensorflow02
cd /d E:\ziliao\LearningPy\face
python face_4.py
```
### hide.vbs文件以隐藏程序运行时的cmd
```
Set ws = CreateObject("Wscript.Shell") 
ws.run "cmd /c E:\ziliao\LearningPy\face\myface.bat",vbhide
``` 
### 添加hide.vbs任务计划库中 
创建任务

常规中|触发器|操作
-|-|-
最高权限 选择对应系统win10|添加 工作站解锁时|添加hide.vbs

# 参考：
* https://www.cnblogs.com/mu---mu/p/FaceRecognition-tensorflow.html
* https://github.com/saksham-jain/Lock-Unlock-Laptop-PC-Screen-Using-Face-Recognition

# CSDN:https://blog.csdn.net/weixin_42348202/article/details/104071199

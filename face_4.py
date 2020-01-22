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
    print("你是张睿祥")
else:
    user32 = windll.LoadLibrary('user32.dll')
    user32.LockWorkStation()


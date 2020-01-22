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
                '''
                if n % 3 == 1:
                    face = relight(face, 1, 50)
                elif n % 3 == 2:
                    face = relight(face, 0.5, 0)
                '''
                # face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
                cv2.imshow('img', face)
                cv2.imwrite(output_dir+'/'+str(index)+'.jpg', face)
                index+=1
            key = cv2.waitKey(30) & 0xff
            if key == 27:
                sys.exit(0)
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

    # restored_model = tf.keras.models.load_model(r'C:\Users\Administrator\Desktop\my_model.h5')
    # pre_result=restored_model.predict_classes(test_x)
    # print(classification_report(test_y, pre_result))

facemain()
# predict_y = model.predict(test_x)


              



# x = K.placeholder(tf.float32, [None, size, size, 3])
# y_ = K.placeholder(tf.float32, [None, 2])

# keep_prob_5 = K.placeholder(tf.float32)
# keep_prob_75 = K.placeholder(tf.float32)

# def weightVariable(shape):
#     init = tf.random_normal(shape, stddev=0.01)
#     return tf.Variable(init)

# def biasVariable(shape):
#     init = tf.random_normal(shape)
#     return tf.Variable(init)

# def conv2d(x, W):
#     return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# def maxPool(x):
#     return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# def dropout(x, keep):
#     return tf.nn.dropout(x, keep)

# def cnnLayer():
#     # 第一层
#     W1 = weightVariable([3,3,3,32]) # 卷积核大小(3,3)， 输入通道(3)， 输出通道(32)
#     b1 = biasVariable([32])
#     # 卷积
#     conv1 = tf.nn.relu(conv2d(x, W1) + b1)
#     # 池化
#     pool1 = maxPool(conv1)
#     # 减少过拟合，随机让某些权重不更新
#     drop1 = dropout(pool1, keep_prob_5)

#     # 第二层
#     W2 = weightVariable([3,3,32,64])
#     b2 = biasVariable([64])
#     conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
#     pool2 = maxPool(conv2)
#     drop2 = dropout(pool2, keep_prob_5)

#     # 第三层
#     W3 = weightVariable([3,3,64,64])
#     b3 = biasVariable([64])
#     conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
#     pool3 = maxPool(conv3)
#     drop3 = dropout(pool3, keep_prob_5)

#     # 全连接层
#     Wf = weightVariable([8*16*32, 512])
#     bf = biasVariable([512])
#     drop3_flat = tf.reshape(drop3, [-1, 8*16*32])
#     dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
#     dropf = dropout(dense, keep_prob_75)

#     # 输出层
#     Wout = weightVariable([512,2])
#     bout = weightVariable([2])
#     #out = tf.matmul(dropf, Wout) + bout
#     out = tf.add(tf.matmul(dropf, Wout), bout)
#     return out

# def cnnTrain():
#     out = cnnLayer()

#     cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_))

#     train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
#     # 比较标签是否相等，再求的所有数的平均值，tf.cast(强制转换类型)
#     accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1)), tf.float32))
#     # 将loss与accuracy保存以供tensorboard使用
#     tf.summary.scalar('loss', cross_entropy)
#     tf.summary.scalar('accuracy', accuracy)
#     merged_summary_op = tf.summary.merge_all()
#     # 数据保存器的初始化
#     saver = tf.train.Saver()

#     with tf.Session() as sess:

#         sess.run(tf.global_variables_initializer())

#         summary_writer = tf.summary.FileWriter('./tmp', graph=tf.get_default_graph())

#         for n in range(10):
#              # 每次取128(batch_size)张图片
#             for i in range(num_batch):
#                 batch_x = train_x[i*batch_size : (i+1)*batch_size]
#                 batch_y = train_y[i*batch_size : (i+1)*batch_size]
#                 # 开始训练数据，同时训练三个变量，返回三个数据
#                 _,loss,summary = sess.run([train_step, cross_entropy, merged_summary_op],
#                                            feed_dict={x:batch_x,y_:batch_y, keep_prob_5:0.5,keep_prob_75:0.75})
#                 summary_writer.add_summary(summary, n*num_batch+i)
#                 # 打印损失
#                 print(n*num_batch+i, loss)

#                 if (n*num_batch+i) % 100 == 0:
#                     # 获取测试数据的准确率
#                     acc = accuracy.eval({x:test_x, y_:test_y, keep_prob_5:1.0, keep_prob_75:1.0})
#                     print(n*num_batch+i, acc)
#                     # 准确率大于0.98时保存并退出
#                     if acc > 0.98 and n > 2:
#                         saver.save(sess, './train_faces.model', global_step=n*num_batch+i)
#                         sys.exit(0)
#         print('accuracy less 0.98, exited!')

# cnnTrain()
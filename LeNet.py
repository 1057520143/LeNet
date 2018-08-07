# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 14:52:43 2018

@author: LiaoWanYi
"""

#coding=utf-8
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.utils.np_utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import plot_model

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((-1,28,28,1))  
y_train = to_categorical(y_train,10)   
x_test = x_test.reshape((-1,28,28,1))  
y_test = to_categorical(y_test,10) 

#model=load_model('E:/Ptest/GoolgeNet/LeNet-5_model.h5')

model = Sequential()

model.add(Conv2D(6,(5,5),strides=(1,1),input_shape=(28,28,1),padding='valid',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(16,(5,5),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(120,activation='relu'))
model.add(Dense(84,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
model.load_weights('E:/Ptest/GoolgeNet/LeNet-5_weigths.h5', by_name=False) 

model.fit(x_train,y_train,batch_size=100,epochs=50,shuffle=True)
model.save('E:/Ptest/GoolgeNet/LeNet-5_model.h5')
#[0.10342620456655367 0.9834000068902969]
loss, accuracy=model.evaluate(x_test, y_test,batch_size=100)
print(loss, accuracy)
plot_model(model, to_file='E:/Ptest/GoolgeNet/model.png',show_shapes=True)

#----------------------------------各个层特征可视化-------------------------------
#查看输入图片
fig1,ax1 = plt.subplots(figsize=(4,4))
ax1.imshow(np.reshape(x_test[12], (28, 28)))
plt.show()

image_arr=np.reshape(x_test[12], (-1,28, 28,1))
#可视化第一个MaxPooling2D
layer_1 = K.function([model.layers[0].input], [model.layers[1].output])
# 只修改inpu_image
f1 = layer_1([image_arr])[0]
# 第一层卷积后的特征图展示，输出是（1,12,12,6），（样本个数，特征图尺寸长，特征图尺寸宽，特征图个数）
re = np.transpose(f1, (0,3,1,2))
for i in range(6):
    plt.subplot(2,4,i+1)
    plt.imshow(re[0][i]) #,cmap='gray'
    
plt.show()
#可视化第二个MaxPooling2D
layer_2 = K.function([model.layers[0].input], [model.layers[3].output])
f2 = layer_2([image_arr])[0]
# 第一层卷积后的特征图展示，输出是（1,4,4,16），（样本个数，特征图尺寸长，特征图尺寸宽，特征图个数）
re = np.transpose(f2, (0,3,1,2))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(re[0][i]) #, cmap='gray'
plt.show() 

#----------------------------------可视化滤波器-------------------------------

#将张量转换成有效图像
def deprocess_image(x):
    # 对张量进行规范化
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    # 转化到RGB数组
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

for i_kernal in range(16):
    input_img=model.input
    ## 构建一个损耗函数，使所考虑的层的第n个滤波器的激活最大化，-1层softmax层
    loss = K.mean(model.layers[3].output[:,:,:,i_kernal])
    # loss = K.mean(model.output[:, :,:, i_kernal])
    # 计算输入图像的梯度与这个损失
    grads = K.gradients(loss, input_img)[0]
    # 效用函数通过其L2范数标准化张量
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    # 此函数返回给定输入图像的损耗和梯度
    iterate = K.function([input_img, K.learning_phase()], [loss, grads])
    # 从带有一些随机噪声的灰色图像开始
    np.random.seed(0)
    #图像通道
    num_channels=1
    #输入图像尺寸
    img_height=img_width=28
    #归一化图像
    input_img_data = (255- np.random.randint(0,255,(1,  img_height, img_width, num_channels))) / 255.
    failed = False
    # run gradient ascent
    print('####################################',i_kernal+1)
    loss_value_pre=0
     # 运行梯度上升500步
    for i in range(1000):
        loss_value, grads_value = iterate([input_img_data,1])
        if i%10 == 0:
            # print(' predictions: ' , np.shape(predictions), np.argmax(predictions))
            print('Iteration %d/%d, loss: %f' % (i, 1000, loss_value))
            print('Mean grad: %f' % np.mean(grads_value))
            
            
            if all(np.abs(grads_val) < 0.000001 for grads_val in grads_value.flatten()):
                failed = True
                print('Failed')
                break
            # print('Image:\n%s' % str(input_img_data[0,0,:,:]))
            if loss_value_pre != 0 and loss_value_pre > loss_value:
                break
            if loss_value_pre == 0:
                loss_value_pre = loss_value

            # if loss_value > 0.99:
            #     break

        input_img_data += grads_value * 1 #e-3
    plt.subplot(4,4, i_kernal+1)
    # plt.imshow((process(input_img_data[0,:,:,0])*255).astype('uint8'), cmap='Greys') #cmap='Greys'
    img_re = deprocess_image(input_img_data[0])
    img_re = np.reshape(img_re, (28,28))
    plt.imshow(img_re) #cmap='Greys'
plt.show()
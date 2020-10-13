# -*- coding: utf-8 -*-
"""
@CNN
"""
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense#, Activation
import matplotlib.pyplot as plt
import numpy as np
#資料讀取
(x_Train,y_Train),(x_Test,y_Test)=mnist.load_data()

##CNN
#資料轉換
x_Train = x_Train.reshape(x_Train.shape[0],28,28,1).astype('float32')
x_Test = x_Test.reshape(x_Test.shape[0],28,28,1).astype('float32')
#Features 標準化
x_Train = x_Train/255
x_Test = x_Test/255

y_Train = np_utils.to_categorical(y_Train)
y_Test = np_utils.to_categorical(y_Test)

#建立模型
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=5,kernel_size=(3,3),input_shape=(28,28,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=10,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=15,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.summary()

#進行訓練
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history = model.fit(x=x_Train,y=y_Train,
                          validation_split=0.2,#抓20%資料來協助模型評估
                          epochs=10,#跑10次
                          batch_size=300)#一次看300筆就更新網路)

print('mean loss:',np.mean(train_history.history["loss"]))
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.title('Train history')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'],loc='best')
plt.show()

print('mean acc:',np.mean(train_history.history["acc"]))
plt.plot(train_history.history['acc'])
plt.plot(train_history.history['val_acc'])
plt.title('Train history')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train','validation'],loc='best')
plt.show()

prediction = model.predict_classes(x_Test)
import pandas as pd
idx=500
for i in range(0,5):
    ax = plt.subplot(1,5,1+i)
    ax.imshow(x_Test[idx].reshape((28,28)),cmap='gray')
    title = "l={},p={}".format(str(np.argmax(y_Test[idx],axis=0)),
                               str(prediction[idx]))
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    idx+=1
plt.show()
print(pd.crosstab(np.argmax(y_Test,axis=1),prediction,
                  rownames=['label'],colnames=['predict']))


##DNN
x_Train = x_Train.reshape(x_Train.shape[0],784).astype('float32')
x_Train = x_Train/255

DNNmodel = Sequential()
DNNmodel.add(Dense(units=10,input_dim=784,activation='relu',name = 'dense_1'))
DNNmodel.add(Dense(units=5,activation='relu',name = 'dense_2'))
DNNmodel.add(Dense(units=10,activation='softmax',name = 'output_layer'))
DNNmodel.summary()

#Training   ##binary_crossentropy常用於二分法
DNNmodel.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history = DNNmodel.fit(x=x_Train,y=y_Train,
                          validation_split=0.2,#抓20%資料來協助模型評估
                          epochs=20,#跑20次
                          batch_size=30,#一次看30比就更新網路
                          verbose=1)#顯示進度條

print('mean loss:',np.mean(train_history.history["loss"]))
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'],loc='upper right')
plt.show()

print('mean acc:',np.mean(train_history.history["acc"]))
plt.plot(train_history.history['acc'])
plt.plot(train_history.history['val_acc'])
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train','validation'],loc='upper right')
plt.show()

















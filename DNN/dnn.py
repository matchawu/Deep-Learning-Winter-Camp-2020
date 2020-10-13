# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 02:41:31 2019

@author: LIU JHIH-CHEN
"""

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

data = pd.read_csv('titanic.csv')

age_mean = data['Age'].mean()
fare_mean = data['Fare'].mean()
data['Age'] = data['Age'].fillna(age_mean)
data['Fare'] = data['Fare'].fillna(fare_mean)
data['Sex'] = data['Sex'].map({'female':0, 'male':1}).astype(int)
Embarked = list(set(data['Embarked']))
data['Embarked'] = data['Embarked'].map({'C':0, 'Q':1, 'S':2}).astype(int)
#標準化
def normalization(data):
    return (data - data.min()) / (data.max() - data.min())

#One hot encoder
def OHE(data):
    x=list(set(data))
    ohe = np.zeros((len(data),len(x)))
    for i in range(len(data)):
        ohe[i][data[i]] = 1
    return ohe

#%%regression
cols = ['Pclass','Sex','Age','SibSp','Parch','Embarked'] 
data1 = data[cols]

data2 = OHE(data1['Embarked'])
df = pd.DataFrame(data2,columns = Embarked)
data3 =pd.concat([data1,df],axis=1)
data0 = data3.drop('Embarked',axis=1)

df_norm = normalization(data0)

#切割資料
train_x = df_norm.iloc[:1000,:]
train_y = normalization(data['Fare'][:1000])

test_x = df_norm.iloc[1000:,:]
test_y = normalization(data['Fare'][1000:])

#%%
#Build model
model = Sequential()

#Input維度=8 /第一層網路node=5 / Activation function為reLU
model.add(Dense(units=5, input_dim=8, activation='relu',name = 'dense_1')) 

#Output維度=1
model.add(Dense(units=1, activation='relu',name = 'output_layer'))

#顯示Model的架構參數量
model.summary()

# Training
model.compile(loss='MSE', optimizer='adam')
train_history = model.fit(x=train_x, y=train_y, 
                          validation_split=0.2, 
                          epochs=50, 
                          batch_size=30,
                          verbose=1)#顯示進度條

print('mean loss:',np.mean(train_history.history["loss"]))
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()         

#evaluate
test_loss = model.evaluate(x=test_x, y=test_y, batch_size=10, verbose=1)

#%%binary
#cols_b = ['Pclass','Sex','Age','SibSp','Parch','Embarked','Fare'] 
cols_b = ['Pclass','Sex','Age','PassengerId','SibSp','Parch','Embarked','Fare'] 

data1b = data[cols_b]

data3b =pd.concat([data1b,df],axis=1)
data0b = data3b.drop('Embarked',axis=1)

df_normb = normalization(data0b)

y = OHE(data['Survived'])

#切割資料
train_xb = df_normb.iloc[:1000,:]
train_yb = y[:1000]

test_xb = df_normb.iloc[1000:,:]
test_yb = y[1000:]
#%%
#Build binary model
model = Sequential()

#Input維度=10 /第一層網路node=5 / Activation function為reLU
model.add(Dense(units=200, input_dim=10, activation='relu',name = 'dense_1')) 

#Output維度=1
model.add(Dense(units=2, activation='softmax',name = 'output_layer'))

#顯示Model的架構參數量
model.summary()

# Training
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history = model.fit(x=train_xb, y=train_yb, 
                          validation_split=0.3, 
                          epochs=50, 
                          batch_size=30,
                          verbose=1)#顯示進度條

print('mean accuracy:',np.mean(train_history.history["acc"]))
plt.plot(train_history.history['acc'])
plt.plot(train_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()    

print('mean loss:',np.mean(train_history.history["loss"]))
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()     

test_loss,test_acc = model.evaluate(x=test_xb, y=test_yb, batch_size=10, verbose=1)    
#%%predict









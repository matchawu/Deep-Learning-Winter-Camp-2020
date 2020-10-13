
import pandas as pd
import numpy as np
from keras.utils import np_utils 

df0 = pd.read_csv('SentimentImage.csv',header=0)
df=df0[:15000]

# split data & Translation of data      
#把切割出來的內容當做一列
data = np.array(df['feature'].str.split(" ",expand=True)).reshape(len(df), 48, 48, 1).astype('float32')
label = np.array(df['label'])

data = data / 255  

label = np_utils.to_categorical(label)

X_Train = data[:(len(data)*4)//5]
X_Test = data[(len(data)*4)//5:]
y_Train = label[:(len(data)*4)//5]
y_Test = label[(len(data)*4)//5:]


from keras.models import Sequential  
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D
  
model = Sequential()

model.add(Conv2D(filters=10,  
                 kernel_size=(3,3),
                 input_shape=(48,48,1),  
                 activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))  
  
model.add(Conv2D(filters=20,  
                 kernel_size=(3,3),   
                 activation='relu'))
  
model.add(MaxPooling2D(pool_size=(2,2)))  

# Add Dropout layer  
#model.add(Dropout(0.25))   
model.add(Flatten()) 
model.add(Dense(100, activation='relu'))
#model.add(Dropout(0.25))
model.add(Dense(7, activation='softmax')) 

model.summary() 

model.compile(loss='categorical_crossentropy',  
              optimizer='adam',  
              metrics=['accuracy'])  

# 開始訓練  
train_history = model.fit(x=X_Train,  
                          y=y_Train, validation_split=0.1,
                          epochs=50, batch_size=1024) 
                        

# 畫出訓練結果 
import matplotlib.pyplot as plt  

plt.plot(train_history.history['acc'])  
plt.plot(train_history.history['val_acc'])  
plt.title('Train History')  
plt.ylabel('acc')  
plt.xlabel('Epoch') 
plt.legend(['train', 'validation'], loc='best')  
plt.show()

plt.plot(train_history.history['loss'])  
plt.plot(train_history.history['val_loss'])  
plt.title('Train History')  
plt.ylabel('loss')  
plt.xlabel('Epoch') 
plt.legend(['train', 'validation'], loc='best') 
plt.show()


# 顯示預測結果
import numpy as np  

prediction = model.predict_classes(X_Test)
idx = 100
for i in range(0, 5):  
    ax = plt.subplot(1,5, 1+i)  
    ax.imshow(X_Test[idx].reshape((48, 48)), cmap='gray') 
    title = "l={},p={}".format(str(np.argmax(y_Test[idx], axis=0)),
                               str(prediction[idx]))     
    ax.set_title(title, fontsize=10)  
    ax.set_xticks([])
    ax.set_yticks([])  
    idx+=1  
plt.show() 

import pandas as pd  
print(pd.crosstab(np.argmax(y_Test, axis=1), prediction, rownames=['label'], colnames=['predict']))


#建立dnn模型
from keras.models import Sequential  
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D  
  
DNNmodel = Sequential()

DNNmodel.add(Flatten(input_shape=(48,48,1)))
DNNmodel.add(Dense(87, activation='relu'))
DNNmodel.add(Dense(25, activation='relu'))  
DNNmodel.add(Dense(7, activation='softmax')) 

DNNmodel.summary() 

DNNmodel.compile(loss='categorical_crossentropy',  
              optimizer='adam',  
              metrics=['accuracy'])  

DNN_train_history = DNNmodel.fit(x=X_Train,  
                          y=y_Train, validation_split=0.1,
                          epochs=100, batch_size=1024) 

# 畫出訓練結果 
import matplotlib.pyplot as plt  

plt.plot(DNN_train_history.history['acc'])  
plt.plot(DNN_train_history.history['val_acc'])  
plt.title('Train History')  
plt.ylabel('acc')  
plt.xlabel('Epoch') 
plt.legend(['train', 'validation'], loc='best')  
plt.show()

plt.plot(DNN_train_history.history['loss'])  
plt.plot(DNN_train_history.history['val_loss'])  
plt.title('Train History')  
plt.ylabel('loss')  
plt.xlabel('Epoch') 
plt.legend(['train', 'validation'], loc='best')  
plt.show()

# 顯示預測結果
import numpy as np  

DNN_prediction = DNNmodel.predict_classes(X_Test)
idx = 100
for i in range(0, 5):  
    ax = plt.subplot(1,5, 1+i)  
    ax.imshow(X_Test[idx].reshape((48, 48)), cmap='gray') 
    title = "l={},p={}".format(str(np.argmax(y_Test[idx], axis=0)),
                               str(DNN_prediction[idx]))     
    ax.set_title(title, fontsize=10)  
    ax.set_xticks([])
    ax.set_yticks([])  
    idx+=1  
plt.show() 

import pandas as pd  
print(pd.crosstab(np.argmax(y_Test, axis=1), DNN_prediction, rownames=['label'], colnames=['predict']))


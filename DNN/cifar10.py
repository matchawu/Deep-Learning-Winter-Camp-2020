# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 16:21:38 2020

@author: Fei-fan He
"""

from keras.datasets import cifar10

((x_train, y_train), (x_test, y_test)) = cifar10.load_data()

#%%
import numpy as np

I = np.eye(10)
y_train = I[y_train.flatten()]
y_test = I[y_test.flatten()]

#%%
x_train = x_train / 255
x_test = x_test / 255

#%%
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

#%%
from keras.models import Model
from keras.layers import Input, Dense

x = Input(shape=(3072,))
z = Dense(200, activation='relu')(x)
o = Dense(10, activation='softmax')(z)

shallow_model = Model(inputs=x, outputs=o)
shallow_model.summary()

#%%
shallow_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history = shallow_model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        batch_size = 500,
        epochs = 50,
        verbose = 1)

#%%
d = Dense(180, activation='relu')(x)
d = Dense(100, activation='relu')(d)
d = Dense(100, activation='relu')(d)
d = Dense(100, activation='relu')(d)
d = Dense(100, activation='relu')(d)
g = Dense(10, activation='softmax')(d)

deep_model = Model(inputs=x, outputs=g)
deep_model.summary()

#%%
deep_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history_deep = deep_model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        batch_size = 500,
        epochs = 50,
        verbose = 1)

#%%
import matplotlib.pyplot as plt

plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.plot(train_history_deep.history['loss'])
plt.plot(train_history_deep.history['val_loss'])
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['shallow train', 'shallow validation', 'deep train', 'deep validation'], loc='upper right')
plt.show()
plt.close()

plt.plot(train_history.history['acc'])
plt.plot(train_history.history['val_acc'])
plt.plot(train_history_deep.history['acc'])
plt.plot(train_history_deep.history['val_acc'])
plt.title('Acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['shallow train', 'shallow validation', 'deep train', 'deep validation'], loc='lower right')
plt.show()
plt.close()
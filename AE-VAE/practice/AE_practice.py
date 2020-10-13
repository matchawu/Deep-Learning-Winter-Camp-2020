# -*- coding: utf-8 -*-
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
#%%
(x_train, _), _ = mnist.load_data()
#%%
#TODO: 取x的0~19999筆為x_train、，做normalization
x_train = 

#%%
#TODO: reshape data使data可以放進model中
x_train =  

#%%
#TODO: 用Conv2D, MaxPooling2D build encoder UpSampling2D build AE model
input_img = Input(shape=(28,28,1))






x = Flatten()(x)
encoded = 
encoder = Model(input_img, encoded)
encoder.summary()
#TODO: 用Conv2D, UpSampling2D build decoder
input_code = Input(shape=(2, ))








decoded = 
decoder = Model(input_code, decoded)
decoder.summary()
#TODO: 把encoder和decoder接在一起
output = 
autoencoder = Model(input_img, output)
autoencoder.summary()

#%%
#TODO: compile and fit model
autoencoder.compile(optimizer='', loss='')
autoencoder.fit(x=, y=, epochs=, batch_size=, shuffle=True, validation_split=0.1)

#%% show result
#TODO: check the distribution of the code for the first 1000 image
encode_code = 
plt.scatter

#TODO: set parameters
unit = 
limit = 

#TODO: 找一個範圍畫圖
xx = np.linspace(-limit, limit, unit)
yy = 
xy = 
plt_xy = np.concatenate([xy[0].reshape((-1, 1)), xy[1].reshape((-1, 1))], axis=1)
decode_img = 

xxyy = np.zeros((unit*28, unit*28))
for j in range(0, unit*28, 28):
    for k in range(0, unit*28, 28):
        xxyy[] = 

plt.imshow(xxyy)
plt.xticks([]) 
plt.yticks([]) 
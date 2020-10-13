# -*- coding: utf-8 -*-
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense

(x_train, _), _ = mnist.load_data()

x_train = x_train/255
x_train = x_train.reshape((x_train.shape[0], -1))

input_img = Input(shape=(x_train.shape[1], ))
x = Dense(128, activation='sigmoid')(input_img)
x = Dense(64, activation='sigmoid')(x)
encoded = Dense(32, activation='sigmoid')(x)
encoder = Model(input_img, encoded)
encoder.summary()

input_code = Input(shape=(32, ))
x = Dense(64, activation='sigmoid')(input_code)
x = Dense(128, activation='sigmoid')(x)
decoded = Dense(784, activation='sigmoid')(x)
decoder = Model(input_code, decoded)
decoder.summary()

output = decoder(encoder(input_img))
autoencoder = Model(input_img, output)	
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=256, batch_size=256, shuffle=True, validation_split=0.1, verbose=1)

decoded_img = autoencoder.predict(x_train)

n=10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_train[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_img[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
# -*- coding: utf-8 -*-
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Reshape

(x_train, _), _ = mnist.load_data()

x_train = x_train[:20000] / 255

x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))


input_img = Input(shape=(28,28,1))
x = Conv2D(16, (3, 3), activation='tanh', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='tanh', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='tanh', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Flatten()(x)
encoded = Dense(2, activation='tanh')(x)
encoder = Model(input_img, encoded)
encoder.summary()

input_code = Input(shape=(2, ))
x = Dense(128, activation='tanh')(input_code)
x = Reshape((4, 4, 8))(x)
x = Conv2D(8, (3, 3), activation='tanh', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='tanh', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='tanh')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
decoder = Model(input_code, decoded)
decoder.summary()

output = decoder(encoder(input_img))
autoencoder = Model(input_img, output)
autoencoder.summary()

autoencoder.compile(optimizer='Adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=64, batch_size=256, shuffle=True, validation_split=0.1)


unit = 15
limit = 1
xx = np.linspace(-limit, limit, unit)
yy = np.linspace(-limit, limit, unit)
xy = np.meshgrid(xx, yy)
plt_xy = np.concatenate([xy[0].reshape((-1, 1)), xy[1].reshape((-1, 1))], axis=1)
decode_img = decoder.predict(plt_xy).reshape((unit*unit, 28, 28))

xxyy = np.zeros((unit*28, unit*28))
for j in range(0, unit*28, 28):
    for k in range(0, unit*28, 28):
        xxyy[j:j+28, k:k+28] = decode_img[j//28+k*unit//28]

plt.imshow(xxyy)
plt.gray()
plt.xticks([]) 
plt.yticks([]) 

plt.figure()
encode_code = encoder.predict(x_train[:1000])
plt.scatter(encode_code[:, 0], encode_code[:, 1])

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
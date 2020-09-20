# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 17:58:44 2020

@author: Calvin Brown
"""

import numpy as np

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from scipy.signal import find_peaks
from matplotlib import pyplot as plt


# load spectral data
spectra = np.load('simulated_spectra.npy')
(n_spectra, n_wavelengths) = spectra.shape

# preprocess and split data
test_frac = 0.1
split_idx = np.round(n_spectra * (1 - test_frac))
X_train = spectra[:split_idx, :]
X_test = spectra[split_idx:, :]
n_train = X_train.shape[0]
n_test = X_test.shape[0]

# create autoencoder model
input_spectrum = Input(shape=(n_wavelengths,))
encoded = Dense(32, activation='relu')(input_spectrum)
decoded = Dense(n_wavelengths, activation='sigmoid')(encoded)
autoencoder = Model(input_spectrum, decoded)

encoder = Model(input_spectrum, encoded)

encoded_input = Input(shape=(32,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

opt = optimizers.Adam(lr=1e-3)
autoencoder.compile(optimizer=opt, loss='binary_crossentropy')

# train model
autoencoder.fit(X_train, X_train, epochs=50, batch_size=256,
                validation_data=(X_test, X_test))

# evaluate model
encoded_spectra = encoder.predict(X_test)
decoded_spectra = decoder.predict(encoded_spectra)

for i in np.random.choice()

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
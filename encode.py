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
wavelengths = np.linspace(400, 700, n_wavelengths)

# preprocess and split data
test_frac = 0.1
split_idx = int(np.round(n_spectra * (1 - test_frac)))
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

for i in np.random.choice(n_test, 5):
    
    fig, ax = plt.subplots()
    ax.plot(wavelengths, X_test[i], label='y')
    ax.plot(wavelengths, decoded_spectra[i, :], label='yhat')
    
    ax.set_xlabel('wavelength (nm)')
    ax.set_ylabel('intensity')
    ax.legend()
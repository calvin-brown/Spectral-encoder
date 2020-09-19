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


# load spectral data
spectra = np.load('simulated_spectra.npy')
(n_spectra, n_wavelengths) = spectra.shape

# preprocess and split data
test_frac = 0.1
split_idx = np.round(n_spectra * (1 - test_frac))
X_train = spectra[:split_idx, :]
X_test = spectra[split_idx:, :]

# create autoencoder model
input_layer = Input(shape=(n_wavelengths,))
encoded = Dense(32, activation='relu')(input_layer)
decoded = Dense(n_wavelengths, activation='sigmoid')(encoded)
autoencoder = Model(input_layer, decoded)

encoder = Model(input_layer, encoded)

encoded_input = Input(shape=(32,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

opt = optimizers.Adam(lr=1e-3)
autoencoder.compile(optimizer=opt, loss='binary_crossentropy')

# train model
autoencoder.fit(X_train, X_train, epochs=50, batch_size=256,
                validation_data=(X_test, X_test))
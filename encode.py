# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 17:58:44 2020

@author: Calvin Brown
"""

import numpy as np

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


# load spectral data
spectra = np.load('simulated_spectra.npy')
(n_spectra, n_wavelengths) = spectra.shape

# create autoencoder model
input_layer = Input(shape=(n_wavelengths,))
encoded = Dense(32, activation='relu')(input_layer)
decoded = Dense(n_wavelengths, activation='sigmoid')(encoded)
autoencoder = Model(input_layer, decoded)

encoder = Model(input_layer, encoded)

encoded_input = Input(shape=(32,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))
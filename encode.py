# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 17:58:44 2020

@author: Calvin Brown
"""

from os.path import join
from datetime import datetime

import numpy as np

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
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
encoder_input = Input(shape=(n_wavelengths,))
x = Dense(512, activation='relu')(encoder_input)
x = Dense(512, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(128, activation='relu')(x)
encoder_output = Dense(16, activation='relu')(x) # 32

encoder = Model(encoder_input, encoder_output)

decoder_input = Input(shape=(16,))
x = Dense(128, activation='relu')(decoder_input)
x = Dense(128, activation='relu')(x)
x = Dense(512, activation='relu')(x)
x = Dense(512, activation='relu')(x)
decoder_output = Dense(n_wavelengths, activation='sigmoid')(x)

decoder = Model(decoder_input, decoder_output)

autoencoder = Model(encoder_input, decoder(encoder(encoder_input)))

opt = optimizers.Adam(lr=1e-4)
autoencoder.compile(optimizer=opt, loss='mse')

# callbacks
earlystopper = EarlyStopping(patience=500)
tensorboard = TensorBoard(join('logs', datetime.now().strftime('%m%d-%H%M%S')))

# train model
autoencoder.fit(X_train, X_train, epochs=50000, batch_size=256,
                validation_data=(X_test, X_test),
                callbacks=[earlystopper, tensorboard])

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
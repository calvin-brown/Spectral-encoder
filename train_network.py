# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 09:48:48 2020

@author: Calvin
"""

import os
from datetime import datetime
from os.path import join

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, TensorBoard
    )

from autoencoder import Autoencoder


# Load spectral data.
spectra = np.load('spectra.npy')
n_spectra, n_wavelengths = spectra.shape
min_wavelength = 400
max_wavelength = 700
wavelengths = np.linspace(min_wavelength, max_wavelength, n_wavelengths)

# Preprocess and split data.
test_frac = 0.01
split_idx = int(n_spectra * (1-test_frac))
X_train = spectra[:split_idx, :]
X_test = spectra[split_idx:, :]
n_train = X_train.shape[0]
n_test = X_test.shape[0]

# Create autoencoder model.
neurons = [128, 32]
latent_size = 16
autoencoder = Autoencoder(neurons, latent_size, n_wavelengths)
opt = optimizers.Adam(lr=1e-4)
autoencoder.compile(optimizer=opt, loss='mse')

# Create callbacks.
filename = datetime.now().strftime('%m%d-%H%M%S')
os.mkdir(join('logs', filename))
os.mkdir(join('networks', filename))
earlystopper = EarlyStopping(patience=100)  # 500)
tensorboard = TensorBoard(join('logs', filename))
checkpointer = ModelCheckpoint(
    filepath=join('networks', filename, 'weights.hdf5'),
    save_best_only=True, save_weights_only=True, verbose=1
    )

# Train model.
autoencoder.fit(
    X_train, X_train, epochs=100000, batch_size=256,
    validation_data=(X_test, X_test),
    callbacks=[earlystopper, tensorboard, checkpointer]
    )

# Reload best model weights.
autoencoder.load_weights(join('networks', filename, 'weights.hdf5'))
autoencoder.compile(optimizer=opt, loss='mse')
print('Loss on test data after reloading best weights')
autoencoder.evaluate(X_test, X_test)

# Save model with best weights.
autoencoder.save(join('networks', filename, 'model'))

# Plot random spectral reconstructions.
decoded_spectra = autoencoder(X_test)
for i in np.random.choice(n_test, 5):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(wavelengths, X_test[i, :], label='y')
    ax.plot(wavelengths, decoded_spectra[i, :], label='yhat')

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity (AU)')
    ax.legend()

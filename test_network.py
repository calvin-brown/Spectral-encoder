# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 10:34:25 2020

@author: Calvin
"""

from os.path import join

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter, sosfiltfilt
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model


filename = '1210-200051'  # [512,64,16], 1e-4, 256. Good, nice step drop
filename = '1210-222215'  # [256,64,16], 1e-4, 256. Better, but no step
filename = '1211-074003'  # [256,64,16], 2e-4, 256. Best and nice step
filename = '1211-101050'  # [256,64,16], 2e-4, 256. Worse, no step
filename = '1211-114513'  # [256,64,16], 2e-4, 256. No step
filename = '1211-130852'  # [256,64,16], 2e-4, 256. 

# Load spectral data. Testing on spectra with narrower peaks (i.e. higher
# spatial frequency content) compared to the training data
spectra = np.load('spectra_narrow.npy')
n_spectra, n_wavelengths = spectra.shape
min_wavelength = 400
max_wavelength = 700
wavelengths = np.linspace(min_wavelength, max_wavelength, n_wavelengths)

# Load trained autoencoder.
autoencoder = load_model(join('networks', filename, 'model'))
opt = optimizers.Adam()
autoencoder.compile(optimizer=opt, loss='mse')
decoded_spectra = autoencoder(spectra)

# Create lowpass filter.
fs = (n_wavelengths-1) / 300  # Sampling freq. Samples per nm
fc = 0.07  # Cutoff freq. Cycles per nm
nyq = 0.5 * fs
fc_norm = fc / nyq  # Cutoff freq normalized by Nyquist freq.
order = 5
sos = butter(order, fc_norm, btype='lowpass', output='sos')

# Plot random spectra.
for i in np.random.choice(n_spectra, 5):
    filtered_spec = sosfiltfilt(sos, spectra[i, :])
    xf = np.arange(n_wavelengths) * fs / n_wavelengths
    xf_shift = (
        np.arange(-n_wavelengths/2, n_wavelengths/2) * (fs/n_wavelengths)
        )

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    fig.set_tight_layout(True)
    for spec, name in zip(
            [spectra[i, :], filtered_spec, decoded_spectra[i, :]],
            ['Raw', 'Filtered', 'Reconstructed']
            ):
        F = np.fft.fft(spec)
        F_shift = np.fft.fftshift(F)
        axs[0].plot(xf_shift, np.abs(F_shift), label=name)
        axs[1].plot(wavelengths, spec, label=name)

    axs[0].set_xlim(-0.2, 0.2)
    axs[0].set_xlabel('Frequency (cycles per nm)')
    axs[0].set_ylabel('Intensity (AU)')
    axs[0].grid()
    axs[0].legend()
    axs[1].set_xlabel('Wavelength (nm)')
    axs[1].set_ylabel('Intensity (AU)')
    axs[1].grid()
    axs[1].legend()

# # FFT tests.
# n = 500  # even
# t = np.linspace(0, 10, n, endpoint=False)  # seconds
# x = np.sin(2*np.pi*15*t) + np.sin(2*np.pi*20*t)  # 15 and 20 Hz
# fig, ax = plt.subplots()
# ax.plot(t, x)
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Intensity (AU)')

# X = np.fft.fft(x)
# X_shift = np.fft.fftshift(X)
# f = np.arange(n) * 50 / n
# f_shift = np.arange(-n/2, n/2) * (50/n)

# fig, ax = plt.subplots()
# ax.plot(f, np.abs(X))

# fig, ax = plt.subplots()
# ax.plot(f_shift, np.abs(X_shift))

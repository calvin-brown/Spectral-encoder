# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 10:34:25 2020

@author: Calvin
"""

from os.path import join

import numpy as np
from scipy.signal import butter, freqz, sosfiltfilt
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from matplotlib import pyplot as plt


filename = '1209-160846'

# Create spectral data.
spectra = np.load('simulated_spectra_gaussian_narrow.npy')
n_spectra, n_wavelengths = spectra.shape
wavelengths = np.linspace(400, 700, n_wavelengths)

# Load autoencoder model.
autoencoder = load_model(join('networks', filename, 'model'))
opt = optimizers.Adam(lr=1e-4)
autoencoder.compile(optimizer=opt, loss='mse', metrics='mse')
# print('Loss on new data after reloading model')
# autoencoder.evaluate(spectra, spectra)
decoded_spectra = autoencoder(spectra)

# Lowpass filter true spectra
fs = (n_wavelengths-1) / 300  # Sampling freq. Samples per nm
fc = 0.1  # Cutoff freq. Cycles per nm
nyq = 0.5 * fs
fc_norm = fc / nyq  # Cutoff freq normalized by Nyquist freq.
order = 3
b, a = butter(order, fc_norm, btype='lowpass')
sos = butter(order, fc_norm, btype='lowpass', output='sos')

for i in np.random.choice(n_spectra, 5):
    filtered_spec = sosfiltfilt(sos, spectra[i, :])
    fig, axs = plt.subplots(2, 1)
    fig.set_tight_layout(True)

    # # Plot the frequency response.
    # w, h = freqz(b, a)
    # axs[0].plot(0.5*fs*w/np.pi, np.abs(h))
    # axs[0].plot(fc, 0.5*np.sqrt(2), 'ko')
    # axs[0].axvline(fc, color='k')
    # axs[0].set_title('Lowpass filter frequency response')
    # axs[0].set_xlabel('Frequency [Hz]')
    # axs[0].grid()

    # Plot Fourier transforms.
    F = np.fft.fft(spectra[i, :])
    F_shift = np.fft.fftshift(F)
    xf = np.arange(n_wavelengths) * fs / n_wavelengths
    xf_shift = (
        np.arange(-n_wavelengths/2, n_wavelengths/2) * (fs/n_wavelengths)
        )
    # xf = np.linspace(0, 0.5*fs, n_wavelengths//2)
    # axs[0].plot(xf, 2/n_wavelengths * np.abs(F[0:n_wavelengths//2]))
    axs[0].plot(xf_shift, np.abs(F_shift), label='Raw')
    F = np.fft.fft(filtered_spec)
    F_shift = np.fft.fftshift(F)
    axs[0].plot(xf_shift, np.abs(F_shift), label='Filtered')
    F = np.fft.fft(decoded_spectra[i, :])
    F_shift = np.fft.fftshift(F)
    axs[0].plot(xf_shift, np.abs(F_shift), label='Reconstructed')
    axs[0].set_xlabel('Frequency (cycles per nm)')
    axs[0].set_ylabel('Intensity (AU)')
    axs[0].grid()
    axs[0].legend()

    # Apply filter.
    axs[1].plot(wavelengths, spectra[i, :], label='Raw')
    axs[1].plot(wavelengths, filtered_spec, label='Filtered')
    axs[1].plot(wavelengths, decoded_spectra[i, :], label='Reconstructed')
    axs[1].set_xlabel('Wavelength (nm)')
    axs[1].set_ylabel('Intensity (AU)')
    axs[1].grid()
    axs[1].legend()

# for i in np.random.choice(n_spectra, 5):
#     fig, ax = plt.subplots()
#     ax.plot(wavelengths, spectra[i], label='y')
#     ax.plot(wavelengths, decoded_spectra[i, :], label='yhat')
#     ax.set_xlabel('wavelength (nm)')
#     ax.set_ylabel('intensity (AU)')
#     ax.legend()

# colors = plt.get_cmap('tab20').colors
# fig, ax = plt.subplots()
# ax.set_prop_cycle(color=colors)
# for i in range(16):
#     encoded = np.zeros((1, 16))
#     encoded[0, i] = 0.01
#     prediction = autoencoder.decoder.predict(encoded)[0]
#     ax.plot(wavelengths, prediction)

# ax.set_xlabel('wavelength (nm)')
# ax.set_ylabel('intensity (AU)')

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

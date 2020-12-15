# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 10:34:25 2020

@author: Calvin
"""

import pickle
from os.path import join

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter, sosfiltfilt
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model


def plot_spectrum(spectrum, decoded_spec):
    filtered_spec = sosfiltfilt(sos, spectrum)
    # xf = np.arange(n_wavelengths) * fs / n_wavelengths
    xf_shift = (
        np.arange(-n_wavelengths/2, n_wavelengths/2) * (fs/n_wavelengths)
        )

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    fig.set_tight_layout(True)
    for spec, name in zip(
            [spectrum, decoded_spec, filtered_spec],
            ['Raw', 'Reconstructed', 'Filtered']
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


filename = '1213-185629'  # [128,32,16], lr 2e-4, bs 256, 1-100 nm.

# Load spectral blind testing data.
spectra = np.load('spectra_test.npy')
n_spectra, n_wavelengths = spectra.shape
min_wavelength = 400
max_wavelength = 700
wavelengths = np.linspace(min_wavelength, max_wavelength, n_wavelengths)
with open('peak_data_test.txt', 'rb') as f:
    peaks = pickle.load(f)

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

# # Plot random spectral reconstructions.
# for i in np.random.choice(n_spectra, 5):
#     plot_spectrum(spectra[i, :], decoded_spectra[i, :])

# # Indices plotted: 2753, 7697, 6599
# for i in [2753, 7697, 6599]:
#     plot_spectrum(spectra[i, :], decoded_spectra[i, :])

# Find indices of narrow spectra.
narrow_idx = []
for i in range(n_spectra):
    if np.max(peaks[i][2]) < 15:
        narrow_idx.append(i)

# Plot random narrow spectra.
for i in np.random.choice(len(narrow_idx), 5):
    plot_spectrum(spectra[narrow_idx[i], :], decoded_spectra[narrow_idx[i], :])

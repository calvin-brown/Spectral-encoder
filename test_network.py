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

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    fig.set_tight_layout(True)
    for spec, name in zip(
            [spectrum, filtered_spec, decoded_spec],
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


filename = '1210-200051'  # [512,64,16], 1e-4, 256. Good, nice step drop
filename = '1210-222215'  # [256,64,16], 1e-4, 256. Better, but no step
filename = '1211-074003'  # [256,64,16], 2e-4, 256. Best and nice step
filename = '1211-101050'  # [256,64,16], 2e-4, 256. Worse, no step
filename = '1211-114513'  # [256,64,16], 2e-4, 256. No step
filename = '1211-130852'  # [256,64,16], 2e-4, 256. No step
filename = '1211-145437'  # [256,64,16], 1e-4, 256, wider. Peaks too wide
filename = '1211-164537'  # [256,64,16], 1e-4, 256, narrower. Bad fits
filename = '1211-173918'  # [256,64,16], 3e-4, 256, narrower. Bad fits
filename = '1211-185601'  # [256,64,16], 1e-4, 256, 5-30nm. okay
filename = '1211-225501'  # [256,64,16], same as above?
filename = '1212-090545'  # [256,64,16], 1e-4, 256, 10-30 MANY.
filename = '1212-145708'  # [256,64,16], 3e-4, 256, 5-100.
filename = '1212-171041'  # [256,32,16], 1e-3, 256, 5-100. GOOD! fits okay and narrow peaks clearly filter

# Load spectral data. Testing on spectra with narrower peaks (i.e. higher
# spatial frequency content) compared to the training data
# spectra = np.load('spectra_narrow.npy')
spectra = np.load('spectra.npy')
n_spectra, n_wavelengths = spectra.shape
min_wavelength = 400
max_wavelength = 700
wavelengths = np.linspace(min_wavelength, max_wavelength, n_wavelengths)
with open('peak_data.txt', 'rb') as f:
    peaks = pickle.load(f)

# Preprocess and split data.
test_frac = 0.01
split_idx = int(n_spectra * (1-test_frac))
X_train = spectra[:split_idx, :]
X_test = spectra[split_idx:, :]
n_train = X_train.shape[0]
n_test = X_test.shape[0]
peaks_test = peaks[split_idx:]

# Load trained autoencoder.
autoencoder = load_model(join('networks', filename, 'model'))
opt = optimizers.Adam()
autoencoder.compile(optimizer=opt, loss='mse')
decoded_spectra = autoencoder(X_test)

# Create lowpass filter.
fs = (n_wavelengths-1) / 300  # Sampling freq. Samples per nm
fc = 0.07  # Cutoff freq. Cycles per nm
nyq = 0.5 * fs
fc_norm = fc / nyq  # Cutoff freq normalized by Nyquist freq.
order = 5
sos = butter(order, fc_norm, btype='lowpass', output='sos')

# # Plot random spectra.
# for i in np.random.choice(n_test, 50):
#     plot_spectrum(X_test[i, :], decoded_spectra[i, :])

# Find narrow spectra.
good_idx = []
for i in range(n_test):
    if np.max(peaks_test[i][2]) < 15:
        good_idx.append(i)

# Plot random narrow spectra.
for i in np.random.choice(len(good_idx), 50):
    plot_spectrum(X_test[good_idx[i], :], decoded_spectra[good_idx[i], :])

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

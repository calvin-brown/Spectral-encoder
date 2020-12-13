# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 09:13:24 2020

@author: Calvin Brown
"""

import pickle

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm


def calculate_std(fwhm):
    return fwhm / 2.355


def create_random_spec():
    curr_spectrum = np.zeros(n_wavelengths)
    n_peaks = np.random.randint(1, max_n_peaks + 1)
    peak_data = [[], [], []]

    for peak in range(n_peaks):
        center = np.random.choice(wavelengths)
        power = np.random.rand()*power_range + min_power
        fwhm = np.random.rand()*fwhm_range + min_fwhm
        # print(f'center {center:.2f} power {power:.2f} fwhm {fwhm:.2f}')
        curr_peak = norm.pdf(wavelengths, center, calculate_std(fwhm))
        curr_peak = curr_peak * (power/np.max(curr_peak))
        curr_spectrum += curr_peak
        peak_data[0].append(center)
        peak_data[1].append(power)
        peak_data[2].append(fwhm)

    if add_noise:
        curr_spectrum += np.random.normal(0, noise_power, n_wavelengths)

    # return np.abs(curr_spectrum), peak_data
    return curr_spectrum, peak_data


# Define constants.
is_narrowband = False
add_noise = True
n_spectra = 1000000
min_wavelength = 400
max_wavelength = 700
n_wavelengths = 1000
wavelengths = np.linspace(min_wavelength, max_wavelength, n_wavelengths)
max_n_peaks = 4
min_power = 0.1
max_power = 1
power_range = max_power - min_power
noise_power = 0.003
if is_narrowband:
    min_fwhm = 5
    max_fwhm = 15
else:
    min_fwhm = 2  # 10
    max_fwhm = 100  # 30
fwhm_range = max_fwhm - min_fwhm

# Generate random spectra.
spectra = np.zeros((n_spectra, n_wavelengths), dtype='float32')
peaks = []
for i in range(n_spectra):
    spectra[i, :], curr_peak_data = create_random_spec()
    peaks.append(curr_peak_data)

np.save(f'spectra{"_narrow" if is_narrowband else ""}', spectra)
with open('peak_data.txt', 'wb') as f:
    pickle.dump(peaks, f)

# Plot random spectra.
fig, ax = plt.subplots(figsize=(9, 5))
for i in np.random.choice(n_spectra, 6):
    ax.plot(wavelengths, spectra[i, :])
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity (AU)')

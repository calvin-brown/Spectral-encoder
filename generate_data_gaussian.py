# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 09:13:24 2020

@author: Calvin Brown
"""

import numpy as np

from scipy.stats import norm
from matplotlib import pyplot as plt


def calculate_std(fwhm):
    return fwhm / 2.355


def create_random_spec():
    curr_spectrum = np.zeros(n_samples)
    n_peaks = np.random.randint(1, max_n_peaks + 1)
    
    for peak in range(n_peaks):
        
        center = np.random.choice(wavelengths)
        power = np.random.rand() * power_range + min_power
        fwhm = np.random.rand() * fwhm_range + min_fwhm
        print(f'center {center:.2f} power {power:.2f} fwhm {fwhm:.2f}')
        
        curr_peak = norm.pdf(wavelengths, center, calculate_std(fwhm))
        curr_peak = curr_peak * (power / np.max(curr_peak))
        curr_spectrum = curr_spectrum + curr_peak
        
    curr_spectrum += np.random.normal(0, noise_power, n_samples)
    return np.abs(curr_spectrum)


# define constants
n_spectra = 100000
min_wavelength = 400
max_wavelength = 700
wavelength_range = max_wavelength - min_wavelength
n_samples = 1000
wavelengths = np.linspace(min_wavelength, max_wavelength, n_samples)
max_n_peaks = 4
min_power = 0.1
max_power = 1
power_range = max_power - min_power
min_fwhm = 10
max_fwhm = 30
fwhm_range = max_fwhm - min_fwhm
noise_power = 0#0.001

# preallocate array of spectra
spectra = np.zeros((n_spectra, n_samples), dtype='float32')

for i in range(n_spectra):
    spectra[i, :] = create_random_spec()

np.save('simulated_spectra_gaussian', spectra)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(wavelengths, spectra[i, :])
ax.set_xlabel('wavelength (nm)')
ax.set_ylabel('intensity (AU)')
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 22:08:44 2020

@author: Calvin
"""

import numpy as np

from scipy.stats import norm
from matplotlib import pyplot as plt


def calculate_std(fwhm):
    
    return fwhm / 2.355


n_spectra = 100000
min_wavelength = 400
max_wavelength = 700
wavelength_range = max_wavelength - min_wavelength
n_samples = 1000
max_n_peaks = 4
min_power = 0.1
max_power = 0.9
power_range = max_power - min_power
min_fwhm = 10
max_fwhm = 20
fwhm_range = max_fwhm - min_fwhm
noise_power = 0.001

# spectra = np.zeros((n_spectra, n_samples), dtype='float32')
# wavelengths = np.linspace(min_wavelength, max_wavelength, n_samples)
spectra = np.zeros((n_spectra, n_samples), dtype=float)
wavelengths = np.linspace(min_wavelength, max_wavelength, n_samples)

for i in range(n_spectra):
    
    curr_spectrum = np.zeros(n_samples)
    n_peaks = np.random.randint(1, max_n_peaks + 1)
    
    for peak in range(n_peaks):
        
        center = np.random.choice(wavelengths)
        power = np.random.rand() * power_range + min_power
        fwhm = np.random.rand() * fwhm_range + min_fwhm
        # print('center', center, 'power', power, 'fwhm', fwhm)
        
        curr_peak = norm.pdf(wavelengths, center, calculate_std(fwhm))
        curr_peak = curr_peak * (power / np.max(curr_peak))
        curr_spectrum = curr_spectrum + curr_peak
        
    curr_spectrum = np.abs(
        curr_spectrum + np.random.normal(0, noise_power, curr_spectrum.shape)
        )
    spectra[i, :] = curr_spectrum

np.save('simulated_spectra', spectra)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(wavelengths, curr_spectrum)
ax.set_xlabel('wavelength (nm)')
ax.set_ylabel('intensity')
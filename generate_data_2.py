# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 18:00:38 2020

@author: Calvin Brown
"""

import numpy as np
from matplotlib import pyplot as plt


n_signals = 100000
signal_length = 1000
noise_power = 0.001
x = np.arange(signal_length)

signals = np.zeros((n_signals, signal_length), dtype='float32')

for i in range(n_signals):
    
    curr_signal = np.zeros(signal_length)
    
    for j in np.arange(1, 33, 1):
        
        # if np.random.rand() > 0.5:
        #     continue
        
        power = np.random.rand()
        phase = np.random.rand() * 2 * np.pi
        
        curr_signal += np.sin(x  * 2 * np.pi * j / signal_length + phase) * power
        
    # curr_signal += np.random.normal(0, noise_power, curr_signal.shape)
    signals[i, :] = curr_signal

np.save('simulated_signals', signals)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(x, curr_signal)
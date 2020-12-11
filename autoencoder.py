# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 16:18:26 2020

@author: Calvin Brown
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model


class Autoencoder(Model):
    def __init__(self, neurons, latent_size, n_wavelengths):
        super(Autoencoder, self).__init__()

        # Encoder
        encoder_neurons = neurons.copy()
        encoder_neurons.append(latent_size)
        encoder_activations = ['relu' for n in encoder_neurons]
        encoder_layers = []
        for n, a in zip(encoder_neurons, encoder_activations):
            encoder_layers.append(Dense(n, activation=a))
        self.encoder = Sequential(encoder_layers)

        # Decoder
        decoder_neurons = neurons.copy()
        decoder_neurons.reverse()
        decoder_activations = ['relu' for n in decoder_neurons]
        decoder_neurons.append(n_wavelengths)
        decoder_activations.append(None)
        decoder_layers = []
        for n, a in zip(decoder_neurons, decoder_activations):
            decoder_layers.append(Dense(n, activation=a))
        self.decoder = Sequential(decoder_layers)

        # self.encoder = Sequential([
        #     Dense(512, activation='relu'),
        #     Dense(512, activation='relu'),
        #     Dense(128, activation='relu'),
        #     Dense(128, activation='relu'),
        #     Dense(16, activation='relu'),
        # ])
        # self.decoder = Sequential([
        #     Dense(128, activation='relu'),
        #     Dense(128, activation='relu'),
        #     Dense(512, activation='relu'),
        #     Dense(512, activation='relu'),
        #     Dense(n_wavelengths)
        # ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# spectral-encoder
This repository explores autoencoders for compressing optical spectra

Typically, optical spectra are captured by a spectrometer and used to identify/quantify the presence of materials with known spectral features or peaks. In this project, we simulate spectra as combinations of Gaussian peaks and other simple waveforms. For sparse spectra, we hope that an autoencoder is able to learn a rather efficient representation, leading to a high rate of compression.

A typical encoder (sans machine learning) might try to encode a signal (in this case a spectrum) in a particular basis in which it is sparse. For instance, simple graphics composed of just a few lines/shapes can be saved in much smaller vector files than in raster image files because they are sparse in the vector object basis.

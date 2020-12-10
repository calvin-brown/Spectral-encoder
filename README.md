# spectral-encoder
## Introduction
This repository explores autoencoders for the compression of one-dimensional signals. Specifically, we have the compression of optical spectra in mind as a target application.

### Autoencoders
Autoencoder networks are a typical 

Typically, optical spectra are captured by a spectrometer and used to identify/quantify the presence of materials with known spectral features or peaks. In this project, we simulate spectra as combinations of Gaussian peaks and other simple waveforms. For sparse spectra (few peaks and features), we hope that an autoencoder is able to learn a rather efficient representation, leading to a high rate of compression.

A typical encoder (sans machine learning) might try to encode a signal (in this case a spectrum) in a particular basis in which it is sparse. For instance, simple graphics composed of just a few lines/shapes can be saved in much smaller vector files than in raster image files because they are sparse in the vector object basis.

In this repo, we explore a few different types of spectra: those that are sparse in the spatial (or "intensity") domain and those that are sparse in the spectral (or frequency) domain. An example of a signal that is sparse in the spatial domain would be a point source of light, and an example of a signal that is sparse in the spectral domain would be a narrow-bandwidth laser

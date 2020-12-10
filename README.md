# spectral-encoder
## Introduction
This repository explores autoencoders for the compression of one-dimensional signals. Specifically, we have the compression of optical spectra in mind as a target application. We will find that the networks learn to essentially lowpass filter the spectra, preserving the general shape of spectral features while sacraficing small, high frequency variations. This aligns well with intuition and in fact with practical compression algorithms.

### Autoencoders
Autoencoder networks are a common type of neural network, with a tell-tale hourglass shape:

![Examples of autoencoders](/images/autoencoder_architectures.png)

The purpose of these networks is to learn to compress the input data (signals, images, etc.), thereby saving memory and/or making data transmission faster. The first half of the network is called the encoder, which learns to convert the input signal into a much more compact representation at the narrow bottleneck in the middle. The second half of the network is the decoder, which takes the output from the encoder and attempts to faithfully reconstruct the original (uncompressed) signal from it.

A typical encoder (sans machine learning) might be designed to leverage knowledge of the expected signal characteristics to encode efficiently. For example, saving a signal in a particular basis in which it is sparse will reduce the memory required. This is why simple graphics composed of just a few lines/shapes can be saved in much smaller vector files than in raster image files because they are sparse in the vector object basis (but not necessarily in the pixel basis).

The hope is that autoencoders can learn more efficient ways of compressing signals than we can design by hand.

### Optical spectra and hyperspectral imaging
Typically, optical spectra are captured by a spectrometer and used to identify/quantify the presence of materials with known spectral features or peaks. In this project, we simulate spectra as combinations of Gaussian peaks with varying intensity and bandwidth (FWHM):

![Examples of simulated spectra](/spectrum1.png)

For sparse spectra (few peaks and features), we hope that an autoencoder is able to learn a rather efficient representation, leading to a high rate of compression.



In this repo, we explore a few different types of spectra: those that are sparse in the spatial (or "intensity") domain and those that are sparse in the spectral (or frequency) domain. An example of a signal that is sparse in the spatial domain would be a point source of light, and an example of a signal that is sparse in the spectral domain would be a narrow-bandwidth laser

# spectral-encoder
## Introduction
This repository explores autoencoders for the compression of one-dimensional signals. Specifically, we have the compression of optical spectra in mind as a target application. We will find that the networks learn to essentially lowpass filter the spectra, preserving the general shape of spectral features while sacrificing small, high frequency variations. This aligns well with intuition and in fact with practical compression algorithms.

### Autoencoders
Autoencoder networks are a common type of neural network, with a tell-tale hourglass shape:

![Examples of autoencoders](/images/autoencoder_architectures.png)

The purpose of these networks is to learn a compressed representation of the input data (signals, images, etc.). This of course can save memory and/or make data transmission faster, but autoencoders are also important for tasks such as image denoising and image generation. The first half of the network is called the encoder, which learns to convert the input signal into a much more compact representation at the narrow bottleneck in the middle. The second half of the network is the decoder, which takes the output from the encoder and attempts to faithfully reconstruct the original (uncompressed) signal from it.

A typical encoder (sans machine learning) might be designed to leverage knowledge of the expected signal characteristics to encode efficiently. For example, saving a signal in a particular basis in which it is sparse will reduce the memory required. This is why simple graphics composed of just a few lines/shapes can be saved in much smaller vector files than in raster image files because they are sparse in the vector object basis (but not necessarily in the pixel basis).

The hope is that autoencoders can learn more efficient ways of compressing signals than we can design by hand.

### Optical spectra and hyperspectral imaging
Typically, optical spectra are captured by a spectrometer and used to identify/quantify the presence of materials with known spectral features or peaks. In this project, we simulate spectra as combinations of Gaussian peaks with varying intensity and bandwidth (FWHM):

![Examples of simulated spectra](/images/spectra.svg)

For sparse spectra (few peaks and features), we hope that an autoencoder is able to learn a rather efficient representation, leading to a high rate of compression. While compression may not be necessary when capturing a spectrum of, e.g., a sample in the lab, it is critical in high-throughput spectroscopy applications. Notably, hyperspectral imaging requires obtaining a spectrum for each pixel in an image, often at 10s of frames per second and maybe mounted on a drone. In these situations, compressing spectra for processing and transmission is much more critical.

## Generating spectra
generate_spectra.py creates spectra by adding a random number of Gaussian peaks with random center wavelength and bandwidth. spectra.npy contains spectra used for training and validation data. spectra_narrow.npy contains spectra (with narrower peaks) used for blind testing. It turns out that shifting the distribution of the testing data towards higher spatial frequencies (i.e. narrower peaks) will demonstrate that the autoencoder network tends to learn to lowpass filter the input spectra.

## Training the network
The autoencoder class is contained in autoencoder.py. train_network.py trains the encoder using an Adam optimizer with a learning rate of 2e-4. The input (and output) spectra contain 1000 points, whereas the latent space contains just 16 nodes, meaning the compression ratio is 1000 / 1.6 = 62.5.  After training, it is clear that the network is able to qualitatively reconstruct unseen spectra quite well:

IMAGE OF VALIDATION SPECTRA RECONSRUCTION

## Testing the network on narrower peaks
test_network.py

# spectral-encoder
## Introduction
This repository explores autoencoders (see [here](https://www.jeremyjordan.me/autoencoders/#:~:text=Autoencoders%20are%20an%20unsupervised%20learning,representation%20of%20the%20original%20input.) and [here](https://www.deeplearningbook.org/contents/autoencoders.html)) for the compression of one-dimensional signals. Specifically, we have the compression of optical spectra in mind as a target application. We will find that the networks can learn to lowpass filter the spectra, preserving the general shape of spectral features while sacrificing small, high frequency variations. This aligns well with intuition and in fact with practical compression algorithms such as JPEG.

NOTE: I use the term "frequency" here to refer to the rate of variation of signal intensity per change in wavelength (1/nm). These are the frequencies on which "lowpass filtering" acts. This does not relate to the frequency of the electromagnetic oscillation of the light, which is ~564 THz for a wavelength of 532 nm.

This repo is loosely related to the ideas discussed in [our recent paper](https://arxiv.org/abs/2012.00878).

### Autoencoders
Autoencoder networks are a common type of neural network, with a tell-tale hourglass shape:

![Examples of autoencoders](/images/autoencoder_architectures.png)

The purpose of these networks is to learn a compressed representation of the input data (signals, images, etc.). This of course can save memory and/or make data transmission faster, but autoencoders are also important for tasks such as image denoising and image generation. The first half of the network is called the encoder, which learns to convert the input signal into a much more compact representation at the narrow bottleneck in the middle. The second half of the network is the decoder, which takes the output from the encoder and attempts to faithfully reconstruct the original (uncompressed) signal from it.

A typical encoder (sans machine learning) might be designed to leverage knowledge of the expected signal characteristics to encode efficiently. For example, saving a signal in a particular basis in which it is sparse will reduce the memory required. This is why simple graphics composed of just a few lines/shapes can be saved in much smaller vector files than in raster image files because they are sparse in the vector object basis (but not necessarily in the pixel basis).

The hope is that autoencoders can learn more efficient ways of compressing signals than we can design by hand.

### Optical spectra and hyperspectral imaging
Typically, optical spectra are captured by a spectrometer and used to identify/quantify the presence of materials with known spectral features or peaks. In this project, we simulate spectra as combinations of Gaussian peaks with varying intensity and bandwidth (FWHM):

![Examples of simulated spectra](/images/sample_spectra.svg)

For sparse spectra (few peaks and features), we hope that an autoencoder is able to learn a rather efficient representation, leading to a high rate of compression. While compression may not be necessary when capturing a spectrum for, e.g., a sample in the lab, it can be more useful in high-throughput spectroscopy applications. Notably, hyperspectral imaging requires obtaining a spectrum for each pixel in an image, often at 10s of frames per second and maybe mounted on a drone. In these situations, compressing spectra for processing and transmission is much more critical.

## Generating spectra
`generate_spectra.py` creates spectra by adding a random number of Gaussian peaks with random center wavelength and bandwidth. `spectra_train.npy` contains spectra used for training and validation data. `spectra_test.npy` contains spectra used for blind testing. The randomly-generated spectra contain peaks with bandwidths from 1 nm (narrower than an LED, broader than a laser) to 100 nm (broadband source).

## Training the network
The autoencoder class is contained in `autoencoder.py`. `train_network.py` trains the encoder using an Adam optimizer with a learning rate of 2e-4. The input (and output) spectra contain 1000 points, whereas the latent space contains just 16 nodes, meaning the compression ratio is 1000 / 1.6 = 62.5.  After training, it is clear that the network is able to qualitatively reconstruct unseen spectra quite well:

![Examples of output on validation spectra](/images/val_spectrum1.svg)
![Examples of output on validation spectra](/images/val_spectrum2.svg)

A TensorBoard log of the training process is contained in the `logs` folder. Because we could afford to generate a large amount of data (1e6 spectra) and our network size was modest (XXX parameters), little, if any overfitting occurs:

![TensorBoard plot of loss curves](/images/log.png)

## Testing the network on narrower peaks
`test_network.py` evaluates the trained autoencoder network on blind testing spectra that were not contained in the train/validation data.

### Broad peaks:

![Examples of output on test spectra](/images/test_spectrum1.svg)

### Broad and narrow peaks:

![Examples of output on test spectra](/images/test_spectrum2.svg)

### Narrow peak:

![Examples of output on test spectra](/images/test_spectrum3.svg)

The fits look good, but there are errors in the flat regions of the spectra, especially right next to strong peaks. Specifically, the reconstructions tend to overshoot and dip below zero on either side of the peaks. These are indications that the autoencoder has essentially learned an approximation of a **lowpass filter.**

## Comparison between trained autoencoder and lowpass filter
Below, I've added a curve to the plots for the input signal after being run through a lowpass filter. The Fourier transform of the spectra is also shown on the top plots. Especially for spectra with narrow peaks (corresponding to a broader range of frequencies), we see that the network output closely matches the lowpass filtered input.

### Broad peaks:

![Examples of output on test spectra](/images/test_spectrum1_fourier.svg)

### Broad and narrow peaks:

![Examples of output on test spectra](/images/test_spectrum2_fourier.svg)

### Narrow peak:

![Examples of output on test spectra](/images/test_spectrum3_fourier.svg)

Clearly, a part of what the autoencoder has learned is that to best reproduce the input signals (minimize mean squared error or MSE), it is most important to preserve the low frequency content at the expense of the higher frequency content. This aligns with the spirit of many compression approaches, for example denoising is a common use for autoencoders and typically involves lowpass filtering the data under the assumption that the high frequency content has a lower signal to noise ratio (SNR) than the low frequency content. The JPEG standard for image compression takes a similar approach. JPEG prioritizes conserving luminance over chrominance (color) information and low spatial frequencies over higher ones. As with our autoencoder, this enables maximum compression without an unacceptable loss in fidelity. In our case we were concerned about the MSE between the input and output spectra, while JPEG is concerned with maintaining image quality for the human visual system.

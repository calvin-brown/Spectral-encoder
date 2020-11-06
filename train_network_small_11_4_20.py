# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 20:34:47 2020

@author: Calvin
"""

import os
import numpy as np
from os.path import join
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from functools import reduce


def load_data(train_runs, test_run, grid_tag):
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    
    for run in train_runs:
        # load X and y
        curr_X = np.load(join(paths[run], 'X_' + str(square_size) + grid_tag + '.npy'))
        curr_y = np.load(join(paths[run], 'y.npy'))
        n_spectra = curr_y.shape[0]
        n_spec_pixels = curr_y.shape[1]
        n_features = curr_X.shape[1]
        
        print('Initial X shape', curr_X.shape)
    
        # determine repeating, saturated/dim spectra
        repeat_idx = []
        repeat_idx.append(np.arange(271)) # initial sweep 480-750
        for i in np.arange(0, n_spectra, 1000): # repeated spectra every 1000
            repeat_idx.append(np.arange(271, 301) + i)
        repeat_idx = np.concatenate(repeat_idx)
        pixel_sat_idx = np.load(join(paths[run], 'pixel_sat_idx.npy'))
        spec_sat_idx = np.load(join(paths[run], 'spec_sat_idx.npy'))
        delete_idx = reduce(np.union1d, (repeat_idx, pixel_sat_idx, spec_sat_idx)) # union of repeated, dim, and saturated spectra
        print('n repeats', len(repeat_idx))
        print('n pixel_sat', len(pixel_sat_idx))
        print('n spec_sat', len(spec_sat_idx))
        
        # delete repeating, saturated/dim spectra from X and y
        curr_X = np.delete(curr_X, delete_idx, axis=0)
        curr_y = np.delete(curr_y, delete_idx, axis=0)
        
        if run == train_runs[-1]: # only use 80% of run 1 for training
            split_idx = np.round(curr_X.shape[0] * 0.8).astype(int)
            X_train.append(curr_X[:split_idx, :])
            y_train.append(curr_y[:split_idx, :])
            X_test.append(curr_X[split_idx:, :])
            y_test.append(curr_y[split_idx:, :])
            print('run ' + str(run) + ': ' + str(curr_y[:split_idx, :].shape[0]) + ' train samples')
            print('run ' + str(run) + ': ' + str(curr_y[split_idx:, :].shape[0]) + ' test samples')
        else:
            X_train.append(curr_X)
            y_train.append(curr_y)
            print('run ' + str(run) + ': ' + str(curr_y.shape[0]) + ' total samples')
        
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    n_spec_pixels = curr_y.shape[1]
    n_features = curr_X.shape[1]
    
    return X_train, y_train, X_test, y_test, n_spec_pixels, n_features


# hyperparams
square_size = 1
use_symm = False
lr = 1e-5
bs = 1024
neurons = [2048,2048,2048,2048]
dropouts = [0.5, 0.5, 0.5, 0.5]
use_bn = True
tag = ''

# prev_network_path = r'C:\Users\Calvin\Documents\DL_spectroscopy_project\09-15\networks\lr1e-05_bs128_[2048, 2048, 2048]_drpt0.03_bnTrue_grid9_decay4000_tr01_v1_20%_history'

paths = [r'C:\Users\Calvin\Documents\DL_spectroscopy_project\09-13',
         r'C:\Users\Calvin\Documents\DL_spectroscopy_project\09-15',
         r'C:\Users\Calvin\Documents\DL_spectroscopy_project\09-18',
         r'C:\Users\Calvin\Documents\DL_spectroscopy_project\09-24']
train_runs = [0, 1]
test_run = 1

if use_symm:
    grid_tag = '_symm'
else:
    grid_tag = ''

# load data
X_train, y_train, X_test, y_test, n_spec_pixels, n_features = load_data(train_runs, test_run, grid_tag)
n_train_samples = X_train.shape[0]
n_test_samples = X_test.shape[0]
print(str(n_train_samples) + ' train, ' + str(n_test_samples) + ' test')

# normalize data
means = np.mean(X_train, axis=0)
stds = np.std(X_train, axis=0)
X_train = (X_train - means) / stds
X_test = (X_test - means) / stds

# convert to float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

for i in range(1):
    
    filename = 'lr' + str(lr) + '_bs' + str(bs) + '_' + str(neurons) + '_drpt' + str(dropouts[-1]) + '_bn' + str(use_bn) + '_grid' + str(square_size) + grid_tag + tag
    os.mkdir(join('logs', filename))
    os.mkdir(join('networks', filename))
    
    # keras initializations
    keras.backend.clear_session()
    
    # # load good network
    # with open(join(prev_network_path, 'model_arch.txt'), 'r') as text_file:
    #     json_string = text_file.read()
    # model = keras.models.model_from_json(json_string)
    # model.load_weights(join(prev_network_path, 'weights_78000.hdf5'))
    
    # opt = optimizers.Adam(lr=lr)
    # model.compile(loss=loss, optimizer=opt)
    
    # create network
    inputs = layers.Input(shape=X_train[0].shape, name='sub-pixel_intensities')
    x = layers.Dense(neurons[0], activation='relu', name='dense_1')(inputs)
    print('x.dtype: %s' % x.dtype.name)
    if use_bn:
        x = layers.BatchNormalization(name='batchnorm_1')(x)
    x = layers.Dropout(dropouts[0], name='dropout_1')(x)
    
    for j in range(1, len(neurons)):
        x = layers.Dense(neurons[j], activation='relu', name='dense_' + str(j + 1))(x)
        if use_bn:
            x = layers.BatchNormalization(name='batchnorm_' + str(j + 1))(x)
        x = layers.Dropout(dropouts[j], name='dropout_' + str(j + 1))(x)
    
    # outputs = layers.Dense(n_spec_pixels, dtype='float32', name='predictions')(x)
    outputs = layers.Dense(n_spec_pixels, name='predictions')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    print('Outputs dtype: %s' % outputs.dtype.name)
    
    # save model architecture
    print(model.summary())
    json_string = model.to_json()
    with open(join('networks', filename, 'model_arch.txt'), "w") as text_file:
        text_file.write(json_string)
    
    # define callbacks
    checkpointer = keras.callbacks.ModelCheckpoint(filepath=join('networks', filename, 'weights.hdf5'), save_best_only=True, save_weights_only=True, verbose=1)
    tensorboard = keras.callbacks.TensorBoard(log_dir=join('logs', filename))
    
    curr_lr = lr
    train_loss = []
    val_loss = []
    
    while curr_lr >= lr / 10:
        
        opt = keras.optimizers.Adam(lr=curr_lr)
        model.compile(loss='mse', optimizer=opt)
        
        history = model.fit(X_train, y_train, batch_size=bs, epochs=1000000, validation_data=(X_test, y_test), callbacks=[checkpointer, tensorboard], verbose=2)
    
        # train_loss.append(history.history['loss'])
        # val_loss.append(history.history['val_loss'])
        curr_lr = curr_lr * np.sqrt(0.1)
        
        # restore best weights
        model.load_weights(join('networks', filename, 'weights.hdf5'))
        
        print('Best val loss so far', model.evaluate(X_test, y_test))
    
# calculate losses
print('calculating loss...')
# train_loss = np.array(train_loss)
# val_loss = np.array(val_loss)
yhat = model.predict(X_test)
mse = np.mean(np.square(yhat - y_test), axis=1)

# mse = []
# for i in range(n_test_samples):
#     mse.append(model.evaluate(np.expand_dims(X_test[i, :], axis=0), np.expand_dims(y_test[i, :], axis=0), verbose=0))
# mse = np.array(mse)
sort_idx = np.argsort(mse)
# np.save(join(paths[train_runs[-1]], 'networks', filename, 'yhat'), yhat) # yhat and loss stay in chron order
# np.save(join(paths[train_runs[-1]], 'networks', filename, 'loss'), mse)
# np.save(join(paths[train_runs[-1]], 'networks', filename, 'history_train_loss'), train_loss)
# np.save(join(paths[train_runs[-1]], 'networks', filename, 'history_val_loss'), val_loss)

n_to_plot = 11
idx = np.linspace(0, n_test_samples-1, n_to_plot).astype(int)
percentiles = np.linspace(100, 0, n_to_plot)
wavelengths = np.linspace(480, 750, n_spec_pixels)
for i in range(n_to_plot):
    
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.plot(wavelengths, y_test[sort_idx[idx[i]], :], label='y', alpha=0.7)
    ax.plot(wavelengths, yhat[sort_idx[idx[i]], :], label='yhat', alpha=0.7)
    curr_diff = yhat[sort_idx[idx[i]], :] - y_test[sort_idx[idx[i]], :]
    ax.plot(wavelengths, curr_diff + 0.5, label='error', alpha=0.7)
    
    ax.set_ylim(-0.05, 0.75)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity')
    ax.set_title(f'Loss: {mse[sort_idx[idx[i]]]:.3e} (' + str(round(percentiles[i])) + ' percentile)')
    ax.legend()
    fig.set_tight_layout(True)
    # fig.savefig(join(paths[train_runs[-1]], 'networks', filename, str(round(percentiles[i])) + 'pct.tiff'))
    # plt.close(fig)


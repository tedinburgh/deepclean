## VARATIONAL AUTOENCODER FOR INTENSIVE CARE UNIT PHYSIOLOGICAL DATA

# ---------------------------------------------------------------------------
# TIME

import time

print('Start: ' + time.ctime())
time_start = time.time()

# ---------------------------------------------------------------------------
# MODULES

from keras import models
from keras import layers
from keras import backend as K
from keras import Input
from keras.losses import mse
from keras.callbacks import EarlyStopping
from keras import optimizers

import h5py
import numpy as np
from scipy.stats import norm
import sys
from os import listdir

# ---------------------------------------------------------------------------
# LOAD DATA

hf = h5py.File('data_input.h5', 'r')

train_data = hf.get('train_data')[()]
validation_data = hf.get('validation_data')[()]
test_data = hf.get('test_data')[()]

hf.close()

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# MODEL

# ---------------------------------------------------------------------------
# PARAMETERS

sample_length = np.shape(train_data)[1]

input_shape = (sample_length, 1, )
batch_size = 32
latent_dim = 5
epochs = 40
patience = 10
Nz = 2
beta = 1

file_suffix = 1
files = listdir()
vae_results_files = np.array(files)[[files[x][:11] == 'vae_results' \
    for x in np.arange(np.size(files))]]
existing_file_suffix = [int(x.strip('vae_results_').strip('.h5')) for x in vae_results_files]
while file_suffix in existing_file_suffix:
    file_suffix += 1

def isInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

if ('--latent_dim' in sys.argv):
    try:
        latent_dim_arg = sys.argv[sys.argv.index('--latent_dim') + 1]
    except:
        latent_dim_arg = None
    if isInt(latent_dim_arg):
        latent_dim = int(latent_dim_arg)
    else:
        err_message = 'latent dim arg input must be an integer'

if ('--beta' in sys.argv):
    try:
        beta_arg = sys.argv[sys.argv.index('--beta') + 1]
    except:
        beta_arg = None
    if isInt(beta_arg):
        beta = int(beta_arg)
    else:
        beta = float(beta_arg)


if ('--file_suffix' in sys.argv):
    try:
        file_suffix = sys.argv[sys.argv.index('--file_suffix') + 1]
    except:
        pass

if ('--learning_rate' in sys.argv):
    try:
        lr_arg = sys.argv[sys.argv.index('--learning_rate') + 1]
        lr = float(lr_arg)
    except:
        lr = 0.001
else:
    lr = 0.001

# ---------------------------------------------------------------------------
# ENCODER

x_input = layers.Input(shape = input_shape)
x = layers.Conv1D(8, 15, padding = 'same', activation = 'relu')(x_input)
shape_mp1 = K.int_shape(x)
x = layers.MaxPooling1D(5, padding = 'same')(x)
x = layers.Conv1D(16, 15, padding = 'same', activation = 'relu')(x)
shape_mp2 = K.int_shape(x)
x = layers.MaxPooling1D(5, padding = 'same')(x)
x = layers.Dropout(rate = 0.1)(x)
x = layers.Conv1D(16, 15, padding = 'same', activation = 'relu')(x)
shape = K.int_shape(x)
x = layers.Flatten()(x)
x = layers.Dropout(rate = 0.1)(x)
x = layers.Dense(16, activation = 'relu')(x)

z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

encoder = models.Model(x_input, [z_mean, z_log_var])
encoder.summary()

# ---------------------------------------------------------------------------
# LATENT SAMPLING

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape = (K.shape(z_mean)[0], latent_dim), \
        mean = 0, stddev = 1)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

latent_mean = layers.Input(shape = (latent_dim, ))
latent_log_var = layers.Input(shape = (latent_dim, ))
z = layers.Lambda(sampling, output_shape = (latent_dim, ))([latent_mean, latent_log_var])
sampler = models.Model([latent_mean, latent_log_var], z)

# ---------------------------------------------------------------------------
# DECODER

latent_input = layers.Input(shape = (latent_dim, ))
x = layers.Dense(16, activation = 'relu')(latent_input)
x = layers.Dense(np.prod(shape[1:]), activation = 'relu')(x)
x = layers.Dropout(rate = 0.1)(x)
x = layers.Reshape(shape[1:])(x)
x = layers.Conv1D(16, 15, padding = 'same', activation = 'relu')(x)
x = layers.UpSampling1D(5)(x)
cropping_up1 = K.int_shape(x)[1] - shape_mp2[1]
x = layers.Cropping1D((0, cropping_up1))(x)
x = layers.Dropout(rate = 0.1)(x)
x = layers.Conv1D(8, 15, padding = 'same', activation = 'relu')(x)
x = layers.UpSampling1D(5)(x)
cropping_up2 = K.int_shape(x)[1] - shape_mp1[1]
x = layers.Cropping1D((0, cropping_up2))(x)
output = layers.Conv1D(1, 15, padding = 'same', activation = None)(x)

decoder = models.Model(latent_input, output)
decoder.summary()

# ---------------------------------------------------------------------------
# VAE

latent_params = encoder(x_input)

x_outputs = [decoder(sampler([latent_params[0], latent_params[1]])) for x in np.arange(Nz)]
x_output = layers.average(x_outputs)

# ---------------------------------------------------------------------------
# LOSS

def reconstruction_loss(x_input, x_outputs):
    reconstruction_loss = K.sum(K.square(x_input[:,:,0] - x_outputs[0][:,:,0]), axis = -1)
    for x in np.arange(1, Nz):
        reconstruction_loss += K.sum(K.square(x_input[:,:,0] - x_outputs[x][:,:,0]), axis = -1)
    reconstruction_loss *= 0.5
    reconstruction_loss /= Nz
    return K.sum(reconstruction_loss)

def kl_loss(z_mean, z_log_var):
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis = -1)
    kl_loss *= -0.5
    kl_loss *= beta
    return K.sum(kl_loss)

class CustomVariationalLayer(layers.Layer):
    def vae_loss(self, x_input, x_outputs, z_mean, z_log_var):
        vae_loss = reconstruction_loss(x_input, x_outputs) + kl_loss(z_mean, z_log_var)
        return vae_loss
    def call(self, inputs):
        x_input = inputs[0]
        x_output = inputs[1]
        z_mean = inputs[2]
        z_log_var = inputs[3]
        x_outputs = inputs[4:]
        loss = self.vae_loss(x_input, x_outputs, z_mean, z_log_var)
        self.add_loss(loss, inputs = inputs)
        return x_output

x_output = CustomVariationalLayer()([x_input, x_output, z_mean, z_log_var] + x_outputs)
vae = models.Model(x_input, x_output)

# ---------------------------------------------------------------------------
# COMPILE

rmsprop = optimizers.RMSprop(lr=lr, rho=0.9, epsilon=None, decay=0.0)

vae.compile(optimizer = rmsprop, loss = None)
vae.summary()

# ---------------------------------------------------------------------------
# METRICS

vae.metrics_tensors.append(reconstruction_loss(x_input, x_outputs))
vae.metrics_names.append('reconstruction_loss')

vae.metrics_tensors.append(kl_loss(z_mean, z_log_var))
vae.metrics_names.append('kl_loss')

# ---------------------------------------------------------------------------

print('Start training: ' + time.ctime())
time_start_train = time.time()

# ---------------------------------------------------------------------------
# FIT

early_stopping = EarlyStopping(monitor = 'val_loss', patience = patience)

history = vae.fit(x = train_data, y = None,
    epochs = epochs,
    batch_size = batch_size,
    validation_data = (validation_data, None),
    callbacks = [early_stopping])

print('Reconstruction loss: ' + str(history.history['reconstruction_loss']))
print('Val reconstruction loss: ' + str(history.history['val_reconstruction_loss']))

print('KL loss: ' + str(history.history['kl_loss']))
print('Val KL loss: ' + str(history.history['val_kl_loss']))

print('Loss: ' + str(history.history['loss']))
print('Val loss: ' + str(history.history['val_loss']))

# ---------------------------------------------------------------------------

print('Training finished: ' + time.ctime())
time_end_train = time.time()

# ---------------------------------------------------------------------------
# WEIGHTS

# vae.save('vae_temp.h5')
# vae.load_weights('vae_temp.h5')

# ---------------------------------------------------------------------------
# LATENT PREDICTION

# Depending on the latent space dimension, it may be unfeasible to sample
#   from a grid embedded in the latent space.

n_grid = 5

max_pages = 25
n_pages = np.min((int(latent_dim * (latent_dim - 1) / 2), max_pages))
n_samples = n_pages * n_grid ** 2

grid = norm.ppf(np.linspace(0.05, 0.95, n_grid))

def to_idx(n, n_grid, latent_dim):
    return (n // n_grid ** np.arange(latent_dim)) % n_grid

idx = np.array([np.flip(to_idx(x, latent_dim, 2), axis = 0) \
    for x in np.arange(latent_dim * (latent_dim - 1))])
idx = np.array([x for x in idx if x[0] < x[1]])
idx = np.repeat(idx, n_grid ** 2, axis = 0)

z_grid = np.repeat(0, latent_dim).astype('float')
z_grid = np.tile(z_grid, (n_samples, 1))

sub_idx = np.array([to_idx(x % n_grid ** 2, n_grid, 2) for x in np.arange(n_grid ** 2)])
sub_idx = np.tile(sub_idx, (int(latent_dim * (latent_dim - 1) / 2), 1))

for x in np.arange(n_samples):
    z_grid[x, idx[x]] = grid[sub_idx[x]]

z_embedding = decoder.predict(z_grid)

# ---------------------------------------------------------------------------
# PREDICTION

z_pred_train = encoder.predict(train_data)
z_pred_val = encoder.predict(validation_data)
z_pred_test = encoder.predict(test_data)

pred_train = decoder.predict(z_pred_train[0])
pred_val = decoder.predict(z_pred_val[0])
pred_test = decoder.predict(z_pred_test[0])

# ---------------------------------------------------------------------------
# WRITE TO FILE

hf = h5py.File('vae_results_' + file_suffix + '.h5', 'w')

hf['train_prediction'] = pred_train
hf['validation_prediction'] = pred_val
hf['test_prediction'] = pred_test

hf['z_train_prediction'] = z_pred_train
hf['z_validation_prediction'] = z_pred_val
hf['z_test_prediction'] = z_pred_test

hf['z_embedding'] = z_embedding
hf['z_grid'] = z_grid

hf['val_loss'] = history.history['val_loss']
hf['loss'] = history.history['loss']

hf['kl_loss'] = history.history['kl_loss']
hf['val_kl_loss'] = history.history['val_kl_loss']
hf['reconstruction_loss'] = history.history['reconstruction_loss']
hf['val_reconstruction_loss'] = history.history['val_reconstruction_loss']

hf['latent_dim'] = latent_dim
hf['beta'] = beta

hf.close()

# ---------------------------------------------------------------------------

print('epochs: ', np.size(history.history['loss']))
print('batch_size: ', batch_size)
print('Nz: ', Nz)
print('latent_dim: ', latent_dim)
print('beta: ', beta)

# ---------------------------------------------------------------------------

print('End: ' + time.ctime())
print('Time to completion: ' + np.str(np.round(time.time() - time_start, 1)) + 's')
print('Time to train: ' + np.str(np.round(time_end_train - time_start_train, 1)) + 's')

# ---------------------------------------------------------------------------


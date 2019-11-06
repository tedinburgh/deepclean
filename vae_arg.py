## VARATIONAL AUTOENCODER FOR INTENSIVE CARE WAVEFORMS

# ---------------------------------------------------------------------------
# MODULES #1

import time
import argparse
import os
import numpy as np

# ---------------------------------------------------------------------------
# TIME

#### DONE

time_start = time.time()
ctime_start = time.ctime()

# ---------------------------------------------------------------------------
# PARSER

#### FIX

parser = argparse.ArgumentParser(description = 'VAE for ICU waveforms')

# FILES
parser.add_argument('--input_file', '-i', \
    type = str, default = 'input.h5', dest = 'input_file', \
    help = 'Input file name')
parser.add_argument('--output_file_suffix', \
    type = str, default = None, dest = 'output_file_suffix', \
    help = 'Output file suffix (identifier)')
parser.add_argument('--output_file', '-o', \
    type = str, default = None, dest = 'output_file', \
    help = 'Output file name (overwrites output_file_suffix)')
parser.add_argument('--output_logfile', '-l', \
    type = bool, default = True, dest = 'output_logfile', \
    help = 'Output logfile printing')
parser.add_argument('--verbose', '-v', \
    action = 'store_true', dest = 'verbose', \
    help = 'Verbose printing')
parser.add_argument('--save_weights', \
    type = str, default = False, dest = 'save_weights', \
    help = 'Save network weights in h5 file (boolean or filename string)')
parser.add_argument('--load_weights', \
    type = str, default = None, dest = 'load_weights', \
    help = 'Load network weights from h5 file (must correspond to network architecture)')

# HYPERPARAMETERS
parser.add_argument('--latent_dim', '-ld', \
    type = int, default = 5, dest = 'latent_dim', \
    help = 'Latent dimension')
parser.add_argument('--Nz', '-Nz', \
    type = int, default = 20, dest = 'Nz', \
    help = 'Number of Monte Carlo samples from latent distribution')

# NETWORK
parser.add_argument('--batch_normalization', '-bn', \
    action = 'store_true', dest = 'batch_normalization', \
    help = 'Batch normalization')
parser.add_argument('--dropout', '-d', \
    type = float, nargs = '*', default = [0.1, 2], dest = 'dropout', \
    help = 'Dropout (rate (first input) on last N (second input) layers of ' + \
        'encoder and first N of decoder); None for no dropout')
parser.add_argument('--enc_network', '-en', \
    type = str, choices = ['fn', 'conv', 'convfn'], default = 'convfn', dest = 'enc_network', \
    help = 'Encoder network type (choose from "fn" (fully connected), "conv" ' + \
    '(convolutional), "convfn" (convolutional with extra fn layer at end))')
parser.add_argument('--dec_network', '-dn', \
    type = str, choices = ['fn', 'conv', 'fnconv'], default = 'fnconv', dest = 'dec_network', \
    help = 'Decoder network type (choose from "fn" (fully connected), "conv" ' + \
    '(convolutional), "convfn" (convolutional with extra fn layer at start))')
parser.add_argument('--enc_layer_size', \
    type = int, nargs = '*', default = [16, 16, 8, 16], dest = 'enc_layer_size', \
    help = 'Encoder layer sizes (minus a final FN layer with size -ld)')
parser.add_argument('--dec_layer_size', \
    type = int, nargs = '*', default = [16, 16, 16, 8], dest = 'dec_layer_size', \
    help = 'Decoder layer sizes (minus a final layer reverting to input size)')
parser.add_argument('--enc_nonlin', \
    type = str, nargs = '*', default = 'relu', dest = 'enc_nonlin', \
    help = 'Encoder non-linearities (if all same, just give once)')
parser.add_argument('--dec_nonlin', \
    type = str, nargs = '*', default = 'relu', dest = 'dec_nonlin', \
    help = 'Decoder non-linearities (if all same, just give once)')
parser.add_argument('--enc_pool_size', \
    type = int, default = [5, 5], dest = 'enc_pool_size', \
    help = 'Pooling sizes (for Max Pooling layers in encoder convolutional networks)')
parser.add_argument('--dec_ups_size', \
    type = int, default = [5, 5], dest = 'dec_ups_size', \
    help = 'Upsampling sizes (for Up Sampling layers in decoder convolutional networks)')
parser.add_argument('--kernel_size', \
    type = int, default = 15, dest = 'kernel_size', \
    help = 'Kernel size (for convolutional layers)')
parser.add_argument('--x_var', '-xv', \
    action = 'store_true', dest = 'x_var', \
    help = 'Compute x (output) variances, else uniformly 1')

# TRAINING
parser.add_argument('--train', '-t', \
    type = bool, default = True, dest = 'train', \
    help = 'Train network (if False and no loaded weights then network weights are random)')
parser.add_argument('--batch_size', \
    type = int, default = 32, dest = 'batch_size', \
    help = 'Batch size')
parser.add_argument('--epochs', \
    type = int, default = 40, dest = 'epochs', \
    help = 'Max training epochs')
parser.add_argument('--patience', \
    type = int, default = 8, dest = 'patience', \
    help = 'Early stopping patience')
parser.add_argument('--optimizer', \
    type = str, choices = ['SGD', 'RMSprop', 'Adam'], default = 'RMSprop', dest = 'optimizer', \
    help = 'Optimizer')
parser.add_argument('--learning_rate', '-lr', \
    type = float, default = 0.001, dest = 'learning_rate', \
    help = 'Learning rate')
# parser.add_argument('--kl_anneal', \
#     action = 'store_true', dest = 'kl_anneal', \
#     help = 'KL annealing')
# parser.add_argument('--temp_start', \
#     type = float, dest = 'temp_start')
# parser.add_argument('--temp_epochs', \
#     type = float, dest = 'temp_epochs')

# PRIOR AND CONDITIONAL MODIFICATIONS
parser.add_argument('--min_z_var', \
    type = float, default = 1e-5, dest = 'min_z_var', \
    help = 'Min z (latent) variance, added to avoid underflow')
parser.add_argument('--min_x_var', \
    type = float, default = 0, dest = 'min_x_var', \
    help = 'Min x (output) variance, added to avoid underflow')

# EXTENSIONS
parser.add_argument('--beta', '-b', \
    type = float, default = 1, dest = 'beta', \
    help = 'beta coefficient')

# parser.add_argument('--extensions')
# parser.add_argument('--activations')
# parser.add_argument('--metrics')

args = parser.parse_args()

# ---------------------------------------------------------------------------
# ARGS

arg_names = list(vars(args).keys())

arg_names_default = \
    [arg for arg in arg_names if np.all(vars(args)[arg] == parser.get_default(arg))]
arg_names_changed = \
    [arg for arg in arg_names if np.any(vars(args)[arg] != parser.get_default(arg))]

if args.output_file_suffix is not None:
    args.output_file_suffix = '_' + args.output_file_suffix

latent_dim = args.latent_dim
Nz = args.Nz

if args.enc_network == 'fn':
    parser.set_defaults(enc_layer_size = [512, 16])
elif args.enc_network == 'conv':
    parser.set_defaults(enc_layer_size = [16, 16, 8])

if args.dec_network == 'fn':
    parser.set_defaults(dec_layer_size = [16, 512])
elif args.dec_network == 'conv':
    parser.set_defaults(dec_layer_size = [8, 16])

enc_size = np.size(args.enc_layer_size)
dec_size = np.size(args.dec_layer_size)

if np.size(args.enc_nonlin) != enc_size:
    repeats = enc_size // np.size(args.enc_nonlin) + 1
    args.enc_nonlin = np.repeat(args.enc_nonlin, repeats)

if np.size(args.dec_nonlin) != dec_size:
    repeats = dec_size // np.size(args.dec_nonlin) + 1
    args.dec_nonlin = np.repeat(args.dec_nonlin, repeats)

if np.size(args.dropout) == 1:
    args.dropout.append(2)

# ---------------------------------------------------------------------------
# VERBOSE

#### DONE

def vprint(text, log = None):
    if args.verbose: print(text)
    if log is not None:
        log.append(text)

log_t = []
log_c = []
log_a = []
log_m = []
log_tr = []

break_str = '\n' + '-' * 50 + '\n'

vprint(break_str)
vprint('Start: ' + ctime_start, log_t)
filename_i, file_extension = os.path.splitext(args.input_file)
file_extension = '.h5'
filename_i = filename_i + file_extension
vprint('Input file: ' + filename_i)
vprint(break_str)
vprint(break_str)

if args.verbose: np.set_printoptions(precision = 3)

vprint('Arguments not at defaults:', log_a)
vprint('', log_a)
for arg in arg_names_changed:
    vprint(arg.upper() + ' (' + arg + ')' + ': value = ' + \
        str(vars(args)[arg]) + ', default = ' + str(parser.get_default(arg)), log_a)
    vprint('Help: ' + parser._option_string_actions['--' + arg].help, log_a)
    vprint('', log_a)
vprint(break_str)
vprint('', log_a)
vprint('Arguments as defaults:', log_a)
vprint('', log_a)
for arg in arg_names_default:
    vprint(arg.upper() + ' (' + arg + ')' + ': value = ' + \
        str(vars(args)[arg]) + ', default = ' + str(parser.get_default(arg)), log_a)
    vprint('Help: ' + parser._option_string_actions['--' + arg].help, log_a)
    vprint('', log_a)
vprint(break_str)

# ---------------------------------------------------------------------------
# MODULES #2

#### DONE

import h5py
import sys
from scipy.stats import norm
import re
import contextlib
import io

f = io.StringIO()

with contextlib.redirect_stdout(f):
    from keras import models
    from keras import layers
    from keras import backend as K
    from keras import Input
    from keras.losses import mse
    from keras.callbacks import EarlyStopping
    from keras import optimizers

s = f.getvalue()
vprint(s)

# ---------------------------------------------------------------------------
# CUSTOM CLASS (FOR DECODER)

class Constant(layers.Layer):
    def __init__(self, constant, **kwargs):
        self.constant = constant
        super(Constant, self).__init__(**kwargs)
    def build(self, input_shape):
        super(Constant, self).build(input_shape)
    def call(self, x):
        return self.constant * K.zeros(input_shape)

# ---------------------------------------------------------------------------
# INPUT / LOAD DATA

#### DONE

hf = h5py.File(filename_i, 'r')

x_train = hf.get('train_data')[()]
x_val = hf.get('validation_data')[()]
x_test = hf.get('test_data')[()]

hf.close()

x_length = np.shape(x_train)[1]
x_channels = np.shape(x_train)[2]

input_shape = (x_length, x_channels, )

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# MODEL

# ---------------------------------------------------------------------------
# ENCODER

#### DONE

x_input = layers.Input(shape = input_shape)

x = x_input
if args.enc_network == 'fn':
    x = layers.Flatten()(x)
    for n in np.arange(enc_size):
        if args.dropout is not None and enc_size - n <= args.dropout[1]:
            x = layers.Dropout(rate = args.dropout[0])(x)
        x = layers.Dense(args.enc_layer_size[n], activation = args.enc_nonlin[n])(x)
        if args.batch_normalization:
            x = layers.BatchNormalization()(x)

elif args.enc_network == 'conv':
    for n in np.arange(enc_size):
        if args.dropout is not None and enc_size - n <= args.dropout[1]:
            x = layers.Dropout(rate = args.dropout[0])(x)
        x = layers.Conv1D(args.enc_layer_size[n], args.kernel_size, \
            padding = 'same', activation = args.enc_nonlin[n])(x)
        if args.batch_normalization:
            x = layers.BatchNormalization(axis = channel_axis)(x)
        if args.enc_pool_size is not None and n < np.size(args.enc_pool_size):
            x = layers.MaxPooling1D(args.enc_pool_size[n], padding = 'same')(x)
    x = layers.Flatten()(x)

elif args.enc_network == 'convfn':
    for n in np.arange(enc_size - 1):
        x = layers.Conv1D(args.enc_layer_size[n], args.kernel_size, \
            padding = 'same', activation = args.enc_nonlin[n])(x)
        if args.batch_normalization:
            x = layers.BatchNormalization(axis = channel_axis)(x)
        if args.enc_pool_size is not None and n < np.size(args.enc_pool_size):
            x = layers.MaxPooling1D(args.enc_pool_size[n], padding = 'same')(x)
        if args.dropout is not None and enc_size - n - 1 <= args.dropout[1]:
            x = layers.Dropout(rate = args.dropout[0])(x)
    x = layers.Flatten()(x)
    x = layers.Dense(args.enc_layer_size[-1], activation = args.enc_nonlin[-1])(x)

z_mean = layers.Dense(args.latent_dim)(x)
z_log_var = layers.Dense(args.latent_dim)(x)
if args.min_z_var != 0:
    z_log_var = layers.Lambda(lambda x: x + K.log(args.min_z_var))(z_log_var)

encoder = models.Model(x_input, [z_mean, z_log_var])

# ---------------------------------------------------------------------------
# VERBOSE

#### DONE

vprint('beta = ' + str(args.beta), log_c)
vprint('Nz = ' + str(Nz), log_c)
vprint('input_shape = ' + str(input_shape), log_c)
vprint('', log_c)
vprint('', log_c)

vprint('# ENCODER', log_c)
vprint('', log_c)
vprint('x_input = layers.Input(shape = input_shape)', log_c)
vprint('', log_c)
vprint('x = x_input', log_c)

if args.enc_network == 'fn':
    vprint('x = layers.Flatten()(x)')
    for n in np.arange(enc_size):
        if args.dropout is not None and enc_size - n <= args.dropout[1]:
            vprint('x = layers.Dropout(rate = ' + str(args.dropout[0]) + ')(x)', log_c)
        vprint('x = layers.Dense(' + str(args.enc_layer_size[n]) + \
            ', activation = "' + args.enc_nonlin[n] + '")(x)', log_c)
        if args.batch_normalization:
            vprint('x = layers.BatchNormalization()(x)', log_c)

elif args.enc_network == 'conv':
    for n in np.arange(enc_size):
        if args.dropout is not None and enc_size - n <= args.dropout[1]:
            vprint('x = layers.Dropout(rate = ' + str(args.dropout[0]) + ')(x)', log_c)
        vprint('x = layers.Conv1D(' + str(args.enc_layer_size[n]) + ', ' + \
            str(args.kernel_size) + ', padding = "same", activation = "' + \
            str(args.enc_nonlin[n]) + '")(x)', log_c)
        if args.batch_normalization:
            vprint('x = layers.BatchNormalization()(x)', log_c)
        if args.enc_pool_size is not None and n < np.size(args.enc_pool_size):
            vprint('x = layers.MaxPooling1D(' + str(args.enc_pool_size[n]) + \
                ', padding = "same")(x)', log_c)
    vprint('x = layers.Flatten()(x)')

elif args.enc_network == 'convfn':
    for n in np.arange(enc_size - 1):
        vprint('x = layers.Conv1D(' + str(args.enc_layer_size[n]) + ', ' + \
            str(args.kernel_size) + ', padding = "same", activation = "' + \
            str(args.enc_nonlin[n]) + '"")(x)', log_c)
        if args.batch_normalization:
            vprint('x = layers.BatchNormalization()(x)', log_c)
        if args.enc_pool_size is not None and n < np.size(args.enc_pool_size):
            vprint('x = layers.MaxPooling1D(' + str(args.enc_pool_size[n]) + \
                ', padding = "same")(x)', log_c)
        if args.dropout is not None and enc_size - n - 1 <= args.dropout[1]:
            vprint('x = layers.Dropout(rate = ' + str(args.dropout[0]) + ')(x)', log_c)
    vprint('x = layers.Flatten()(x)')
    vprint('x = layers.Dense(' + str(args.enc_layer_size[-1]) + \
        ', activation = ' + str(args.enc_nonlin[-1]) + ')(x)', log_c)

vprint('', log_c)
vprint('z_mean = layers.Dense(' + str(latent_dim) + ')(x)', log_c)
vprint('z_log_var = layers.Dense(' + str(latent_dim) + ')(x)', log_c)
if args.min_z_var != 0:
    vprint('z_log_var = layers.Lambda(lambda x: x + ' + \
        str(np.log(args.min_z_var)) + ')(z_log_var)', log_c)
vprint('', log_c)
vprint('encoder = models.Model(x_input, [z_mean, z_log_var])', log_c)
vprint('', log_c)

# ---------------------------------------------------------------------------
# Z-SAMPLING

def sampling(args):
    mean, log_var = args
    epsilon = K.random_normal(shape = (K.shape(mean)[0], dim), \
        mean = 0, stddev = 1)
    return mean + K.exp(log_var) * epsilon

z_mean_input = layers.Input(shape = (latent_dim, ))
z_log_var_input = layers.Input(shape = (latent_dim, ))
dim = latent_dim
z = layers.Lambda(sampling, output_shape = (latent_dim, ))([z_mean_input, z_log_var_input])
z_sampler = models.Model([z_mean_input, z_log_var_input], z)

# ---------------------------------------------------------------------------
# VERBOSE

vprint('# Z-SAMPLING', log_c)
vprint('', log_c)

vprint('def sampling(args):', log_c)
vprint('\tmean, log_var = args', log_c)
vprint('\tepsilon = K.random_normal(shape = (K.shape(mean)[0], dim), \\', log_c)
vprint('\t\tmean = 0, stddev = 1)', log_c)
vprint('\treturn mean + K.exp(log_var) * epsilon', log_c)

vprint('', log_c)
vprint('z_mean_input = layers.Input(shape = (latent_dim, ))', log_c)
vprint('z_log_var_input = layers.Input(shape = (latent_dim, ))', log_c)
vprint('dim = latent_dim', log_c)
vprint('z = layers.Lambda(sampling, output_shape = (latent_dim, ))([z_mean_input, \
    z_log_var_input])', log_c)
vprint('z_sampler = models.Model([z_mean_input, z_log_var_input], z)', log_c)
vprint('', log_c)

# ---------------------------------------------------------------------------
# DECODER

#### DONE

z_input = layers.Input(shape = (latent_dim, ))

if not args.x_var:
    n_branches = 1
if args.x_var:
    n_branches = 2

for m in np.arange(n_branches):
    x = z_input

    if args.dec_network == 'fn':
        for n in np.arange(dec_size):
            x = layers.Dense(args.dec_layer_size[n], activation = args.dec_nonlin[n])(x)
            if args.batch_normalization:
                x = layers.BatchNormalization()(x)
            if args.dropout is not None and n <= args.dropout[1]:
                x = layers.Dropout(rate = args.dropout[0])(x)
        x = layers.Dense(x_length * x_channels)(x)

    elif args.dec_network == 'conv':
        size = np.ceil(x_length / np.prod(args.dec_ups_size)).astype(int)
        shape = (size, args.dec_layer_size[0])
        size_prev = size
        x = layers.Dense(np.prod(shape), activation = args.dec_nonlin[0])(x)
        x = layers.Reshape(shape)(x)
        for n in np.arange(1, dec_size):
            x = layers.Conv1D(args.dec_layer_size[n], args.kernel_size, \
                padding = 'same', activation = args.dec_nonlin[n])(x)
            if args.dec_ups_size is not None and n - 1 < np.size(args.dec_ups_size):
                x = layers.UpSampling1D(args.dec_ups_size[n - 1])(x)
                size = np.ceil(x_length / \
                    np.prod(args.dec_ups_size[:np.size(args.dec_ups_size) - n])).astype(int)
                shape_amend = size_prev * args.dec_ups_size[n - 1] - size
                size_prev = size
                if shape_amend != 0:
                    x = layers.Cropping1D((0, shape_amend))(x)
        x = layers.Conv1D(x_channels, args.kernel_size, padding = 'same', activation = None)(x)

    elif args.dec_network == 'fnconv':
        x = layers.Dense(args.dec_layer_size[0], activation = args.dec_nonlin[0])(x)
        size = np.ceil(x_length / np.prod(args.dec_ups_size)).astype(int)
        shape = (size, args.dec_layer_size[1])
        size_prev = size
        x = layers.Dense(np.prod(shape), activation = args.dec_nonlin[1])(x)
        x = layers.Reshape(shape)(x)
        for n in np.arange(2, dec_size):
            x = layers.Conv1D(args.dec_layer_size[n], args.kernel_size, \
                padding = 'same', activation = args.dec_nonlin[n])(x)
            if args.dec_ups_size is not None and n - 2 < np.size(args.dec_ups_size):
                x = layers.UpSampling1D(args.dec_ups_size[n - 2])(x)
                size = np.ceil(x_length / \
                    np.prod(args.dec_ups_size[:np.size(args.dec_ups_size) - n - 1])).astype(int)
                shape_amend = size_prev * args.dec_ups_size[n - 2] - size
                size_prev = size
                if shape_amend != 0:
                    x = layers.Cropping1D((0, shape_amend))(x)
        x = layers.Conv1D(x_channels, args.kernel_size, padding = 'same', activation = None)(x)

    if m == 0:
        x_mean = layers.Reshape((x_length, x_channels))(x)
    if m == 0 and n_branches == 1:
        x_log_var = Constant(constant = 0)(z_input)
    elif m == 1:
        x_log_var = layers.Reshape((x_length, x_channels))(x)

decoder = models.Model(z_input, [x_mean, x_log_var])

# ---------------------------------------------------------------------------
# VERBOSE

## DONE

vprint('# DECODER', log_c)
vprint('', log_c)
vprint('z_input = layers.Input(shape = (' + str(latent_dim) + ', ))', log_c)
vprint('', log_c)

for m in np.arange(n_branches):
    vprint('x = z_input', log_c)

    if args.dec_network == 'fn':
        for n in np.arange(dec_size):
            vprint('x = layers.Dense(' + str(args.dec_layer_size[n]) + \
                ', activation = ' + str(args.dec_nonlin[n]) + ')(x)', log_c)
            if args.batch_normalization:
                vprint('x = layers.BatchNormalization()(x)', log_c)
            if args.dropout is not None and n < args.dropout[1]:
                vprint('x = layers.Dropout(rate = ' + str(args.dropout[0]) + ')(x)', log_c)
        vprint('x = layers.Dense(' + str(x_length * x_channels) + ')(x)', log_c)

    elif args.dec_network == 'conv':
        size = np.ceil(x_length / np.prod(args.dec_ups_size)).astype(int)
        shape = (size, args.dec_layer_size[0])
        size_prev = size
        vprint('x = layers.Dense(' + str(np.prod(shape)) + ', activation = "' + \
            str(args.dec_nonlin[0]) + '")(x)', log_c)
        vprint('x = layers.Reshape(' + str(shape) + ')(x)')
        for n in np.arange(1, dec_size):
            vprint('x = layers.Conv1D(' + str(args.dec_layer_size[n]) + ', ' + \
                str(args.kernel_size) + ', padding = "same", activation = "' + \
                str(args.dec_nonlin[n]) + '")(x)', log_c)
            if args.dec_ups_size is not None and n - 1 < np.size(args.dec_ups_size):
                vprint('x = layers.UpSampling1D(' + str(args.dec_ups_size[n - 1]) + ')(x)', log_c)
                size = np.ceil(x_length / \
                    np.prod(args.dec_ups_size[:np.size(args.dec_ups_size) - n])).astype(int)
                shape_amend = size_prev * args.dec_ups_size[n - 1] - size
                size_prev = size
                if shape_amend != 0:
                    vprint('x = layers.Cropping1D((0, ' + str(shape_amend) + '))(x)', log_c)
        vprint('x = layers.Conv1D(' + str(x_channels) + ', ' + \
            str(args.kernel_size) + ', padding = "same", activation = None)(x)')

    elif args.dec_network == 'fnconv':
        vprint('x = layers.Dense(' + str(args.dec_layer_size[0]) + \
            ', activation = "' + str(args.dec_nonlin[0]) + '")(x)', log_c)
        size = np.ceil(x_length / np.prod(args.dec_ups_size)).astype(int)
        shape = (size, args.dec_layer_size[0])
        size_prev = size
        vprint('x = layers.Dense(' + str(np.prod(shape)) + ', activation = "' + \
            str(args.dec_nonlin[0]) + '")(x)', log_c)
        vprint('x = layers.Reshape(' + str(shape) + ')(x)')
        for n in np.arange(2, dec_size):
            vprint('x = layers.Conv1D(' + str(args.dec_layer_size[n]) + ', ' + \
                str(args.kernel_size) + ', padding = "same", activation = "' + \
                str(args.dec_nonlin[n]) + '")(x)', log_c)
            if args.dec_ups_size is not None and n - 2 < np.size(args.dec_ups_size):
                vprint('x = layers.UpSampling1D(' + str(args.dec_ups_size[n - 2]) + ')(x)', log_c)
                size = np.ceil(x_length / \
                    np.prod(args.dec_ups_size[:np.size(args.dec_ups_size) - n - 1])).astype(int)
                shape_amend = size_prev * args.dec_ups_size[n - 2] - size
                size_prev = size
                if shape_amend != 0:
                    vprint('x = layers.Cropping1D((0, ' + str(shape_amend) + '))(x)', log_c)
        vprint('x = layers.Conv1D(' + str(x_channels) + ', ' + \
            str(args.kernel_size) + ', padding = "same", activation = None)(x)')

    if m == 0:
        vprint('x_mean = layers.Reshape((' + str(x_length) + ', ' + \
            str(x_channels) + '))(x)', log_c)
        vprint('', log_c)
    if m == 0 and n_branches == 1:
        vprint('x_log_var = Constant(constant = 0)(z_input)')
        vprint('', log_c)
    elif m == 1:
        vprint('x_log_var = layers.Reshape((' + str(x_length) + ', ' + \
            str(x_channels) + '))(x)', log_c)
        vprint('', log_c)

vprint('decoder = models.Model(z_input, [x_mean, x_log_var])', log_c)
vprint('', log_c)

# ---------------------------------------------------------------------------
# OUTPUT

#### DONE

z_params = encoder(x_input)
dim = latent_dim
x_params = [decoder(z_sampler([z_params[0], z_params[1]])) for z in np.arange(Nz)]
x_means = [x_params[x][0] for x in np.arange(Nz)]
x_log_vars = [x_params[x][1] for x in np.arange(Nz)]
if args.min_x_var != 0:   
    x_log_vars = [layers.Lambda(lambda y: y + np.log(args.min_x_var))(x_log_vars[x]) \
        for x in np.arange(Nz)]

# ---------------------------------------------------------------------------
# VERBOSE

vprint('# OUTPUT: ', log_c)
vprint('', log_c)
vprint('z_params = encoder(x_input)', log_c)
vprint('dim = latent_dim', log_c)
vprint('x_params = [decoder(z_sampler([z_params[0], z_params[1]])) ' + \
    'for z in np.arange(Nz)]', log_c)
vprint('x_means = [x_params[x][0] for x in np.arange(Nz)]', log_c)
vprint('x_log_vars = [x_params[x][1] for x in np.arange(Nz)]', log_c)

# ---------------------------------------------------------------------------
# LOSS

#### DONE

def reconstruction_loss(x_input, x_means, x_log_vars):
    reconstruction_loss = x_length * K.log(2 * np.pi) + K.sum(x_log_vars[0] + \
        K.square(x_input - x_means[0]) / K.exp(x_log_vars[0]), axis = -1)
    for x in np.arange(1, Nz):
        reconstruction_loss += x_length * K.log(2 * np.pi) + K.sum(x_log_vars[x] + \
            K.square(x_input - x_means[x]) / K.exp(x_log_vars[x]), axis = -1)
    reconstruction_loss *= 0.5
    reconstruction_loss /= Nz
    return K.sum(reconstruction_loss)

def kl_loss(z_mean, z_log_var):
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis = -1)
    kl_loss *= -0.5
    return K.sum(kl_loss)

class AddLoss(layers.Layer):
    def vae_loss(self, x_input, x_means, x_log_vars, z_mean, z_log_var, beta):
        vae_loss = reconstruction_loss(x_input, x_means, x_log_vars) + \
            kl_loss(z_mean, z_log_var) * beta
        return vae_loss
    def call(self, inputs):
        z_mean = inputs[0]
        z_log_var = inputs[1]
        x_input = inputs[2]
        x_means = inputs[3:3 + Nz]
        x_log_vars = inputs[3 + Nz:]
        beta = args.beta
        loss = self.vae_loss(x_input, x_means, x_log_vars, z_mean, z_log_var, beta)
        self.add_loss(loss, inputs = inputs)
        return x_input

x = AddLoss()([z_mean, z_log_var, x_input] + x_means + x_log_vars)
vae = models.Model(x_input, x)

# ---------------------------------------------------------------------------
# VERBOSE / CODE

vprint('# LOSS: COPY AND PASTE LOSS FUNCTIONS AND CLASS FROM SCRIPT', log_c)
vprint('', log_c)
vprint('x = AddLoss()([z_mean, z_log_var, x_input] + x_means + x_log_vars)', log_c)
vprint('vae = models.Model(x_input, x)', log_c)
vprint('', log_c)

# ---------------------------------------------------------------------------
# COMPILE / OPTIMISER

#### DONE

if args.optimizer is 'SGD':
    optimizer = optimizers.SGD(lr = args.learning_rate)
elif args.optimizer is 'RMSprop':
    optimizer = optimizers.RMSprop(lr = args.learning_rate)
elif args.optimizer is 'Adam':
    optimizer = optimizers.Adam(lr = args.learning_rate)

vae.compile(optimizer = optimizer, loss = None)

# ---------------------------------------------------------------------------
# VERBOSE

vprint('')
vprint('# COMPILING: ')
vprint('')
vprint('beta: ' + str(args.beta))
vprint('Optimizer: ' + args.optimizer)

# ---------------------------------------------------------------------------
# METRICS

#### FIX

vae.metrics_tensors.append(reconstruction_loss(x_input, x_means, x_log_vars))
vae.metrics_names.append('reconstruction_loss')

vae.metrics_tensors.append(kl_loss(z_mean, z_log_var))
vae.metrics_names.append('kl_loss')

# ---------------------------------------------------------------------------
# WEIGHTS

#### DONE

if not args.load_weights == False:
    filename_wi = args.load_weights
    vprint('Loading weights, file: ' + filename_wi)
    vprint(break_str)
    vae.load_weights(filename_w)

# ---------------------------------------------------------------------------
# TIME

#### DONE

time_train = time.time()
ctime_train = time.ctime()

# ---------------------------------------------------------------------------
# VERBOSE

#### DONE

vprint(break_str + 'Encoder:')
encoder.summary(print_fn = lambda x: vprint(x, log_m))
vprint('Decoder:')
decoder.summary(print_fn = lambda x: vprint(x, log_m))
vprint(break_str + 'Full model:')
vae.summary(print_fn = lambda x: vprint(x, log_m))
vprint(break_str)

if args.train:
    vprint('Start training: ' + ctime_train, log_t)
else:
    vprint('No training', log_t)
vprint('Time since start: ' + \
    np.str(np.round(time_train - time_start, 1)) + 's', log_t)
vprint(break_str)

# ---------------------------------------------------------------------------
# FIT

#### DONE

early_stopping = EarlyStopping(monitor = 'val_loss', patience = args.patience)

if args.train:
    f = io.StringIO()

    with contextlib.redirect_stoud(f):
        history = vae.fit(x = train_data, y = None,
            epochs = args.epochs,
            batch_size = args.batch_size,
            validation_data = (validation_data, None),
            callbacks = [early_stopping])

    s = f.getvalue()
    vprint(s, log_tr)

epochs = np.size(history.history['loss'])

# ---------------------------------------------------------------------------
# TIME

#### DONE

time_endtrain = time.time()
ctime_endtrain = time.ctime()

# ---------------------------------------------------------------------------
# WEIGHTS

#### DONE

if not args.save_weights == False:
    if args.save_weights:
        filename_wo = 'weights' + args.output_file_suffix + '.h5'
    else:
        filename_wo, file_extension = os.path.splitext(args.save_weights)
        file_extension = '.h5'
        filename_wo = filename_w + file_extension
    vae.save_weights(filename_w)

# ---------------------------------------------------------------------------
# VERBOSE

#### DONE

def metric_name_fun(string, KL_bool = True):
    string = ' '.join(s for s in string.split('_'))
    string = string[0].upper() + string[1:]
    if KL_bool:
        string = ' ' + string
        string = re.sub(' kl ', ' KL ', string)
        string = re.sub(' Kl ', ' KL ', string)
        string = string[1:]
    return string

if args.train:
    for metric in vae.metrics_names:
        vprint(mname_fun(metric))
        vprint(str(history.history[metric] + '\n'))
        vprint(mname_fun('Val ' + metric))
        vprint(str(history.history['val_' + metric] + '\n'))
    vprint(break_str)

    vprint('Finished training: ' + ctime_endtrain, log_t)
    vprint('Time since start: ' + \
        np.str(np.round(time_endtrain - time_start, 1)) + 's', log_t)
    vprint(break_str)
    vprint('Number of epochs in training: ' + epochs, log_tr)
    vprint(break_str)

if not args.save_weights == False:
    vprint('Saving weights, file: ' + filename_wo)
    vprint(break_str)

# ---------------------------------------------------------------------------
# PREDICTION

#### DONE

z_pred_train = encoder.predict(train_data)
z_pred_val = encoder.predict(validation_data)
z_pred_test = encoder.predict(test_data)

pred_train = decoder.predict(z_pred_train[0])
pred_val = decoder.predict(z_pred_val[0])
pred_test = decoder.predict(z_pred_test[0])

# ---------------------------------------------------------------------------
# OUTPUT / WRITE TO FILE

#### DONE

if args.output_file is not None:
    filename_o, file_extension = os.path.splitext(args.output_file)
    file_extension = '.h5'
    filename_o = filename_o + file_extension
else:
    filename_o = 'output' + args.output_file_suffix + '.h5'

results = h5py.File(filename_o, 'w')

results['train_prediction'] = pred_train
results['validation_prediction'] = pred_val
results['test_prediction'] = pred_test

results['z_train_prediction'] = z_pred_train
results['z_validation_prediction'] = z_pred_val
results['z_test_prediction'] = z_pred_test

for metric in vae.metrics_names:
    results[metric] = history.history[metric]
    results['val_' + metric] = history.history['val_' + metric]

results.close()

# ---------------------------------------------------------------------------
# TIME

#### DONE

time_end = time.time()
ctime_end = time.ctime()

# ---------------------------------------------------------------------------
# VERBOSE

#### DONE

vprint('Saving results, file: ' + filename_o)

vprint(break_str)
vprint('End: ' + ctime_end, log_t)
vprint('Time to completion: ' + \
    np.str(np.round(time_end - time_start, 1)) + 's', log_t)
vprint('Time to train: ' + \
    np.str(np.round(time_end_train - time_start_train, 1)) + 's', log_t)
vprint(break_str)

# ---------------------------------------------------------------------------
# LOG

#### DONE

filename_lo = 'log' + args.output_file_suffix + '.log'

break_str = '#' * 50

log = []
log.append(break_str)
log.append(break_str)
log.append('')
log.append('LOGFILE')
log.append('')
log.append(break_str)
log.append(break_str)

log.append('TIME')
for l in log_t:
    log.append(l)
log.append('\n' + break_str)

log.append('FILES')
log.append('Output file: ' + filename_o)
if not args.load_weights == False:
    log.append('Input weights file: ' + filename_wi)
if not args.load_weights == False:
    log.append('Output weights file: ' + filename_wo)
log.append('Output log file: ' + filename_lo)
log.append('\n' + break_str)

log.append('CODE')
for l in log_c:
    log.append(l)
log.append('\n' + break_str)

log.append('ARGUMENTS')
for l in log_a:
    log.append(l)
log.append('\n' + break_str)

log.append('MODELS')
for l in log_m:
    log.append(l)
log.append('\n' + break_str)

log.append('TRAINING')
for l in log_tr:
    log.append(l)
log.append('\n' + break_str)

with open(filename_lo, 'w') as f:
    for l in log:
        f.write(l + '\n')

# ---------------------------------------------------------------------------
# COVARIANCE NETWORK

# sparsity_distance = 5
# n_components = int((sparsity_distance + 1) * (sample_length - sparsity_distance / 2))
# n_offdiag = int(sparsity_distance * (sample_length - (sparsity_distance + 1)/ 2))

# x = layers.Dense(64, activation = 'relu')(latent_input)
# L_log_diag = layers.Dense(sample_length, activation = None)(x)
# L_diag = K.exp(L_log_diag)

# x = layers.Dense(sample_length, activation = 'relu')(x)
# x = layers.Dropout(rate = 0.1)(x)
# L_offdiag = layers.Dense(n_offdiag, activation = None)(x)

# def LTx_fun(L_diag, L_offdiag, x):
#     y = L_diag * x
#     kk = 0
#     for ii in np.arange(offdiag_dist) + 1:
#         y[:-ii] += L_offdiag[kk:kk + sample_length - ii] * x[ii:]
#         kk = kk + sample_length - ii
#     return y

# def log_detSigma(L_log_diag):
#     return -2 * K.sum(L_log_diag)

# def LTinvx_fun(L_diag, L_offdiag, x):
#     y = np.zeros_like(x)
#     y[-1] = x[-1] / L_diag[-1]
#     pos_add = np.cumsum(np.flip(np.arange(-offdiag_dist, 0), axis = 0) + 1 + sample_length)
#     for ii in np.arange(1, sample_length) + 1:
#         jj = sample_length - ii + 1
#         pos = pos_add - ii
#         pos = pos[:(sample_length - jj)]
#         z = L_offdiag[pos]
#         y[-ii] = (x[-ii] - y[jj:jj + np.size(pos)].dot(z)) / L_diag[-ii]
#     return y

# def LTinvx_fun(L_diag, L_offdiag, u):
#     y = K.zeros_like(u)
#     y[:,-1] = u[:,-1] / L_diag[:,-1]
#     pos_add = np.cumsum(np.flip(np.arange(-offdiag_dist, 0), axis = 0) + 1 + sample_length)
#     for ii in np.arange(1, sample_length) + 1:
#         jj = sample_length - ii + 1
#         pos = pos_add - ii
#         pos = pos[:(sample_length - jj)]
#         z = L_offdiag[pos]
#         y[-ii] = (u[-ii] - y[jj:jj + np.size(pos)].dot(z)) / L_diag[-ii]
#     return y

# x = layers.Dense(64, activation = 'relu')(latent_input)
# L_log_diag = layers.Dense(sample_length, activation = None)(x)
# L_diag = K.exp(L_log_diag)
# x = layers.Dense(sample_length, activation = 'relu')(x)
# x = layers.Dropout(rate = 0.1)(x)
# L_offdiag = layers.Dense(n_offdiag, activation = None)(x)
# L_values = K.tf.concat([L_diag, L_offdiag], axis = 1)

# # x = layers.Dense(64, activation = 'relu')(latent_input)
# # x = layers.Dense(sample_length, activation = 'relu')(x)
# # x = layers.Dropout(rate = 0.1)(x)
# # L_values = layers.Dense(n_components, activation = None)(x)

# L = K.zeros((K.shape(x_input)[0], sample_length, sample_length))

# mask = np.array([np.arange(sample_length), np.arange(sample_length)])
# for ii in np.arange(sparsity_distance) + 1:
#     mask = np.append(mask, [np.arange(ii, sample_length), np.arange(sample_length - ii)], axis = 1)
# mask = mask.astype('int32')

# t = K.tf.constant([[[1, 1, 1], [2, 2, 2]],
#                  [[3, 3, 3], [4, 4, 4]],
#                  [[5, 5, 5], [6, 6, 6]]])

# K.tf.boolean_mask()

# L = K.tf.contrib.distributions.fill_triangular(L_values)

# L = K.tf.matrix_set_diag(L, K.exp(K.tf.matrix_diag(L)))

# u = K.random_normal(shape = (K.shape(x_input)[0], sample_length), \
#         mean = 0, stddev = 1)

# eps = K.tf.matrix_triangular_solve(K.transpose(L), u, lower = True)

# # eps = LTinvx_fun(L_diag, L_offdiag, u)

# uncertainty = models.Model(latent_input, eps)
# uncertainty.summary()


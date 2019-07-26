## VARATIONAL AUTOENCODER FOR INTENSIVE CARE UNIT PHYSIOLOGICAL DATA

# ---------------------------------------------------------------------------
## modules

import numpy as np
import matplotlib.pyplot as plt
import time
import h5py
import os

from sklearn.decomposition import PCA

import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection, PolyCollection

# ---------------------------------------------------------------------------
## ABP waveform data - load labels, input

# CHANGE AS REQUIRED
file_path = '/Users'
file_name_input = 'data_input.h5'
file_name_labels = 'labels.h5'

try:
    hf_input = h5py.File(file_path + '/' + file_name_input, 'r')
    train_data = hf_input['train_data'][()]
    validation_data = hf_input['validation_data'][()]
    test_data = hf_input['test_data'][()]
    shuffle = hf_input['shuffle'][()]
    standardisation = hf_input['standardisation'][()]
    mean = standardisation['mean'][0]
    std = standardisation['std'][0]
    test_idx = hf_input['test_idx'][()]
    t_params = hf_input['t_params'][()]
    t_interval = t_params['t_interval'][0]
    sample_length = t_params['sample_length'][0]
    hf_input.close()
except:
    print('Input file missing or not correctly named')

try:
    hf_label = h5py.File('labels.h5', 'r')
    test_label_s = hf_label['test_label_sample'][()]
    test_label_ws = hf_label['test_label_withinsample'][()]
    hf_label.close()
except:
    print('Label file missing or not correctly named')

dmin = np.min(test_data)
dmax = np.max(test_data)

n_test = np.shape(test_data)[0]
n_train = np.shape(train_data)[0]

t = np.arange(sample_length) * t_interval / sample_length

# ---------------------------------------------------------------------------
## test labels

p_artefact_ws = np.sum(test_label_ws) / np.size(test_label_ws)
p_artefact_s = np.sum(test_label_s) / np.size(test_label_s)

test_ws_idx = np.hstack(np.sum(test_label_ws, axis = 1) != 0)

n_test_ws_idx = np.sum(test_ws_idx)

# ---------------------------------------------------------------------------
## display with labels

def plot_poly(t, data, label, **kwargs):
    sample_length = np.size(data)

    if 'col' in kwargs.keys():
        col = kwargs['col']
    else:
        col = plt.get_cmap('tab10').colors[0]

    if 'alpha' in kwargs.keys():
        alpha = kwargs['alpha']
    else:
        alpha = 0.2

    verts = np.zeros((sample_length - 1, 4, 2))
    for idx in np.arange(sample_length - 1):
        ts = np.array([t[idx], t[idx + 1], t[idx + 1], t[idx]])
        ys = np.array([dmin, dmin, dmax, dmax])
        verts[idx] = list(zip(ts, ys))

    poly_col = np.tile([1, 1, 1], (sample_length, 1)).astype(float)
    if np.size(np.where(label)) > 0:
        poly_col[np.where(label), :] = col

    poly = PolyCollection(verts, facecolors = poly_col, edgecolors = None)

    poly.set_alpha(alpha)
    plt.gca().add_collection(poly)

with PdfPages('display_test_label_ws.pdf') as pdf:

    fig = plt.figure(figsize = (20, 25))

    for ii in np.arange(n_test):
        ll = ii % 12

        ax = plt.subplot(4, 3, ll + 1)
        plt.plot(t, test_data[ii], color = (0, 0, 0))
        plot_poly(t, test_data[ii], test_label_ws[ii])
        ax.set_ylim([dmin, dmax])

        if ii % 12 == 11:
            pdf.savefig()
            plt.close()
            fig = plt.figure(figsize = (20, 25))

    pdf.savefig()  # saves the current figure into a pdf page
    plt.close('all')

with PdfPages('display_test_label_s.pdf') as pdf:

    fig = plt.figure(figsize = (20, 25))

    colors = plt.get_cmap('tab10').colors

    for ii in np.arange(n_test):
        ll = ii % 12

        ax = plt.subplot(4, 3, ll + 1)
        plt.plot(t, test_data[ii], color = (0, 0, 0))

        if test_label_s[ii] == 1:
            verts = np.zeros((1, 4, 2))
            ts = np.array([t[0], t[-1], t[-1], t[0]])
            ys = np.array([dmin, dmin, dmax, dmax])
            verts[0] = list(zip(ts, ys))
            poly = PolyCollection(verts, facecolors = colors[0], \
                edgecolors = None)
            poly.set_alpha(0.2)
            ax.add_collection(poly)

        ax.set_ylim([dmin, dmax])

        if ii % 12 == 11:
            pdf.savefig()
            plt.close()
            fig = plt.figure(figsize = (20, 25))

    pdf.savefig()
    plt.close('all')

# ---------------------------------------------------------------------------
## latent dimensions

latent_dims = [2, 3, 4, 5, 10, 20, 50, 100]
n_dims = np.size(latent_dims)

# ---------------------------------------------------------------------------
## pca reconstruction

pca = PCA(n_components = latent_dims[-1])
pca.fit(train_data[:,:,0])

pca_components = pca.components_.copy()

train_reconstruction_pca = np.zeros((n_train, sample_length, n_dims))
test_reconstruction_pca = np.zeros((n_test, sample_length, n_dims))

for ii in np.arange(n_dims):
    n_components = latent_dims[ii]
    pca.components_ = pca_components[:n_components]
    train_reconstruction_pca[:,:,ii] = pca.inverse_transform(pca.transform(train_data[:,:,0]))
    test_reconstruction_pca[:,:,ii] = pca.inverse_transform(pca.transform(test_data[:,:,0]))

# ---------------------------------------------------------------------------
## load vae reconstructions

file_dir_vae = 'vae_files'
file_prefix = 'vae_results_'

files = os.listdir('./' + file_dir_vae)
vae_files = np.array(files)[[files[x][:np.size(file_prefix)] == file_prefix \
    for x in np.arange(np.size(files))]]

train_reconstruction_vae = np.zeros_like(train_reconstruction_pca)
test_reconstruction_vae = np.zeros_like(test_reconstruction_pca)

def read_var_slurm(slurm_suffix, var):
    with open(file_dir_vae + '/slurm-' + str(slurm_suffix) + '.out') as slurm_file:
        for line in slurm_file.readlines()[-10:]:
            if var in line:
                value = int(line.strip(var).strip(':').strip(' '))
                return value

for vae_file in vae_files:
    with h5py.File(file_dir_vae + '/' + vae_file, 'r') as hf_vae:
        if 'latent_dim' in list(hf_vae.keys()):
            latent_dim_file = hf_vae['latent_dim'][()]
        else:
            try:
                suffix = int(vae_file.strip(file_prefix).strip('.h5'))
                latent_dim_file = read_var_slurm(suffix, 'latent_dim')
            except:
                pass
        ii = np.where(latent_dim_file == latent_dims)[0][0]
        train_reconstruction_vae[:,:,ii] = hf_vae['train_prediction'][:,:,0]
        test_reconstruction_vae[:,:,ii] = hf_vae['test_prediction'][:,:,0]

# ---------------------------------------------------------------------------
## reconstruction for example training sample

with PdfPages('reconstruction.pdf') as pdf:
    
    fig = plt.figure(figsize = (11, 5))
    gs = gridspec.GridSpec(4, 11)
    
    colors = plt.get_cmap('Dark2').colors

    fig_latent_dims = [2, 5, 10, 100]

    ax_data = plt.subplot(gs[1:3, :2])
    plt.plot(t, train_data[0, :, 0], color = (0, 0, 0), label = 'Data', lw = 1.3)
    ax_data.set_ylabel('ABP (mmHg)')
    ax_data.set_xlabel('Time (s)')
    ax_data.set_title('Example data')

    for ii in np.flip(np.arange(4)):
        ax_current = plt.subplot(gs[:2, 2 * ii + 3:2 * ii + 5], \
            sharex = ax_data, sharey = ax_data)
        plt.plot(t, train_data[0], color = (0, 0, 0, 0.4), lw = 1)
        jj = np.where(fig_latent_dims[ii] == np.array(latent_dims))[0]
        plt.plot(t, train_reconstruction_pca[0, :, jj][0, :], \
            color = colors[ii], lw = 1.3)
        plt.setp(ax_current.get_xticklabels(), visible = False)
        plt.setp(ax_current.get_yticklabels(), visible = False)
        ax_current.set_title('Latent dim ' + str(latent_dims_fig[ii]))

    ax_current.set_ylabel('PCA reconstruction\nABP (mmHg)')
    plt.setp(ax_current.get_yticklabels(), visible = True)

    for ii in np.flip(np.arange(4)):
        ax_current = plt.subplot(gs[2:, 2 * ii + 3:2 * ii + 5], \
            sharex = ax_data, sharey = ax_data)
        plt.plot(t, train_data[0], color = (0, 0, 0, 0.4), lw = 1)
        jj = np.where(fig_latent_dims[ii] == np.array(latent_dims))[0]
        plt.plot(t, train_reconstruction_vae[0, :, jj][0, :], \
            color = colors[ii], lw = 1.3)
        plt.setp(ax_current.get_yticklabels(), visible = False)

    ax_current.set_ylabel('VAE reconstruction\nABP (mmHg)')
    plt.setp(ax_current.get_yticklabels(), visible = True)

    gs.tight_layout(fig)
    gs.update(wspace = 0, hspace = 0)

    pdf.savefig()
    plt.close()

# ---------------------------------------------------------------------------
## mse

def mse(x, y, axis): 
     return np.square(np.subtract(x, y)).mean(axis = axis)

train_mse_pca = np.zeros((n_train, n_dims))
test_mse_pca = np.zeros((n_test, n_dims))

for ii in np.arange(n_dims):
    train_mse_pca[:,ii] = np.array([mse(train_data[x,:,0], \
        train_reconstruction_pca[x,:,ii], 0) for x in np.arange(n_train)])
    test_mse_pca[:,ii] = np.array([mse(test_data[x,:,0], \
        test_reconstruction_pca[x,:,ii], 0) for x in np.arange(n_test)])


train_mse_vae = np.zeros((n_train, n_dims))
test_mse_vae = np.zeros((n_test, n_dims))

for ii in np.arange(n_dims):
    train_mse_vae[:,ii] = np.array([mse(train_data[x,:,0], \
        train_reconstruction_vae[x,:,ii], 0) for x in np.arange(n_train)])
    test_mse_vae[:,ii] = np.array([mse(test_data[x,:,0], \
        test_reconstruction_vae[x,:,ii], 0) for x in np.arange(n_test)])

# ---------------------------------------------------------------------------
## log-mse thresholds

threshold_percentile = 90

threshold_mse_pca = np.percentile(train_mse_pca, threshold_percentile, axis = 0)
threshold_mse_vae = np.percentile(train_mse_vae, threshold_percentile, axis = 0)

# ---------------------------------------------------------------------------
## tp, tn, fp, fn (global)

def tp_fun(truth, prediction):
    return np.sum(np.logical_and(truth, prediction))
def tn_fun(truth, prediction):
    return np.sum(np.logical_and(np.logical_not(truth), np.logical_not(prediction)))
def fn_fun(truth, prediction):
    return np.sum(np.logical_and(truth, np.logical_not(prediction)))
def fp_fun(truth, prediction):
    return np.sum(np.logical_and(np.logical_not(truth), prediction))

tp_pca = np.array([tp_fun(test_label_s, test_mse_pca[:,ii] > threshold_mse_pca[ii]) \
    for ii in np.arange(n_dims)])
tn_pca = np.array([tn_fun(test_label_s, test_mse_pca[:,ii] > threshold_mse_pca[ii]) \
    for ii in np.arange(n_dims)])
fp_pca = np.array([fp_fun(test_label_s, test_mse_pca[:,ii] > threshold_mse_pca[ii]) \
    for ii in np.arange(n_dims)])
fn_pca = np.array([fn_fun(test_label_s, test_mse_pca[:,ii] > threshold_mse_pca[ii]) \
    for ii in np.arange(n_dims)])

tp_vae = np.array([tp_fun(test_label_s, test_mse_vae[:,ii] > threshold_mse_vae[ii]) \
    for ii in np.arange(n_dims)])
tn_vae = np.array([tn_fun(test_label_s, test_mse_vae[:,ii] > threshold_mse_vae[ii]) \
    for ii in np.arange(n_dims)])
fp_vae = np.array([fp_fun(test_label_s, test_mse_vae[:,ii] > threshold_mse_vae[ii]) \
    for ii in np.arange(n_dims)])
fn_vae = np.array([fn_fun(test_label_s, test_mse_vae[:,ii] > threshold_mse_vae[ii]) \
    for ii in np.arange(n_dims)])

acc_pca = (tp_pca + tn_pca) / (tp_pca + tn_pca + fp_pca + fn_pca)
sen_pca = tp_pca / (tp_pca + fn_pca)
spe_pca = tn_pca / (fp_pca + tn_pca)

acc_vae = (tp_vae + tn_vae) / (tp_vae + tn_vae + fp_vae + fn_vae)
sen_vae = tp_vae / (tp_vae + fn_vae)
spe_vae = tn_vae / (fp_vae + tn_vae)

table = np.stack((np.append(acc_pca, np.mean(acc_pca)), \
    np.append(acc_vae, np.mean(acc_vae)), \
    np.append(sen_pca, np.mean(sen_pca)), \
    np.append(sen_vae, np.mean(sen_vae)), \
    np.append(spe_pca, np.mean(spe_pca)), \
    np.append(spe_vae, np.mean(spe_vae))))

# ---------------------------------------------------------------------------
## logmse figure

with PdfPages('logmse.pdf') as pdf:

    def train_random_width(x, ii):
        percentile = np.where(np.logical_and(x > train_percentiles[:-1, ii], \
            x <= train_percentiles[1:, ii]))[0]
        if x == train_percentiles[0, ii]:
            percentile = 0
        return train_percentiles_diff[percentile, ii]
        
    colors = plt.get_cmap('tab10').colors
    test_colors = [colors[x] for x in test_label_s.astype(int)] 

    fig = plt.figure(figsize = (12, 6))
    gs = gridspec.GridSpec(1, 2)

    # PCA plot
    ax_pca = plt.subplot(gs[0])

    n_percentiles = 51
    train_percentiles = np.percentile(train_mse_pca, np.arange(0, 102, 2), axis = 0)
    width = 1 / np.abs(np.diff(np.log(train_percentiles), axis = 0))        

    for ii in np.arange(n_dims):
        verts = np.zeros((n_percentiles - 1, 4, 2))
        width[:, ii] /= np.max(width[:, ii])
        for pt in np.arange(n_percentiles - 1):
            ys = np.array([train_percentiles[pt, ii], train_percentiles[pt + 1, ii], \
                train_percentiles[pt + 1, ii], train_percentiles[pt, ii]])
            xs = np.array([-1, -1, 1, 1]) * width[pt, ii] * 0.1 + ii
            verts[pt] = list(zip(xs, ys))
        poly = PolyCollection(verts[1:-1], facecolors = (0, 0, 0), \
            edgecolors = (0, 0, 0))
        ax_pca.add_collection(poly)

        ind_pca = np.where(np.logical_or(train_mse_pca[:,ii] < train_percentiles[1, ii], \
            train_mse_pca[:,ii] > train_percentiles[-2, ii]))
        plt.scatter(np.repeat(ii, np.size(ind_pca)), \
            np.log(train_mse_pca[ind_pca, ii]), color = (0, 0, 0), s = 1.5, lw = 0)
        plt.scatter(ii + 0.2 * (2 - test_label_s + 0.5 * np.random.random(size = n_test)), \
            np.log(test_mse_pca[:,ii]), color = test_colors, s = 3, lw = 0)
        plt.plot([ii - 0.15, ii + 0.65], \
            np.log([threshold_mse_pca[ii], threshold_mse_pca[ii]]), \
            color = (0.7, 0.1, 0.1))

    ax_pca.set_xticks(np.arange(n_dims) + 0.2)
    ax_pca.set_xticklabels(latent_dims)

    ax_pca.set_ylabel('log-MSE')
    ax_pca.set_xlabel('Latent dimension')

    plt.title('PCA')

    # VAE plot
    ax_vae = plt.subplot(gs[1], sharey = ax_pca, sharex = ax_pca)

    train_percentiles = np.percentile(train_mse_vae, np.arange(0, 102, 2), axis = 0)
    width = 1 / np.abs(np.diff(np.log(train_percentiles), axis = 0))

    for ii in np.arange(n_latent_dims):
        verts = np.zeros((n_percentiles - 1, 4, 2))
        width[:, ii] /= np.max(width[:, ii])
        for pt in np.arange(n_percentiles - 1):
            ys = np.array([train_percentiles[pt, ii], train_percentiles[pt + 1, ii], \
                train_percentiles[pt + 1, ii], train_percentiles[pt, ii]])
            xs = np.array([-1, -1, 1, 1]) * width[pt, ii] * 0.1 + ii
            verts[pt] = list(zip(xs, ys))
        poly = PolyCollection(verts[1:-1], facecolors = (0, 0, 0), \
            edgecolors = (0, 0, 0))
        ax_vae.add_collection(poly)

        ind_vae = np.where(np.logical_or(train_mse_vae[:,ii] < train_percentiles[1, ii], \
            train_mse_vae[:,ii] > train_percentiles[-2, ii]))
        plt.scatter(np.repeat(ii, np.size(ind_vae)), \
            np.log(train_mse_vae[ind_vae, ii]), color = (0, 0, 0), s = 1.5, lw = 0)
        plt.scatter(ii + 0.2 * (2 - test_label_s + 0.5 * np.random.random(size = n_test)), \
            np.log(test_mse_vae[:,ii]), color = test_colors, s = 3, lw = 0)
        plt.plot([ii - 0.15, ii + 0.65], \
            np.log([threshold_mse_vae[ii], threshold_mse_vae[ii]]), \
            color = (0.7, 0.1, 0.1))

    ax_vae.set_xlabel('Latent dimension')
    plt.title('VAE')

    plt.setp(ax_vae.get_yticklabels(), visible=False)

    # legend
    training = mlines.Line2D([], [], marker = 'o', color = (0, 0, 0), \
        lw = 0, label = 'Training')
    test_artefact = mlines.Line2D([], [], marker = 'o', color = colors[1], \
        lw = 0, label = 'Test (artefact)')
    test_none = mlines.Line2D([], [], marker = 'o', color = colors[0], \
        lw = 0, label = 'Test (no artefact)')
    threshold = mlines.Line2D([], [], color = (0.7, 0.1, 0.1), label = 'log-MSE threshold')
    plt.legend(loc = 'lower right', handles = [training, test_artefact, test_none, threshold])

    plt.suptitle('Sample log-MSE')

    gs.tight_layout(fig)
    gs.update(wspace = 0, hspace = 0)

    pdf.savefig()
    plt.close()

# ---------------------------------------------------------------------------
## rolling window mse

def window_mse(x, y, window):
    sample_length = np.size(x)
    half_window = np.ceil(window / 2).astype(int)

    def start_idx(jj):
        return np.max((jj - half_window, 0))
    def end_idx(jj):
        return np.min((jj + window - half_window, sample_length))

    z = np.array([mse(x[start_idx(jj):end_idx(jj)], y[start_idx(jj):end_idx(jj)], axis = 0) \
        for jj in np.arange(sample_length)])
    return z

t_window = 0.5

window = np.ceil(t_window * sample_length / t_interval).astype(int)

test_wmse_pca = np.zeros_like(test_reconstruction_pca)
test_wmse_vae = np.zeros_like(test_reconstruction_vae)

for ii in np.arange(n_dims):
    test_wmse_pca[:,:,ii] = \
        np.array([window_mse(test_data[x,:,0], test_reconstruction_pca[x,:,ii], window) \
            for x in np.arange(n_test)])
    test_wmse_vae[:,:,ii] = \
        np.array([window_mse(bad_data[x,:,0], test_reconstruction_vae[x,:,ii], window) \
            for x in np.arange(n_test)])

# THIS IS INCREDIBLY TIME CONSUMING!
train_wmse_pca = np.zeros_like(train_reconstruction_pca)
train_wmse_vae = np.zeros_like(train_reconstruction_vae)

for ii in np.arange(n_dims):
    train_wmse_pca[:,:,ii] = \
        np.array([window_mse(train_data[x,:,0], train_reconstruction_pca[x,:,ii], window) \
            for x in np.arange(n_train)]):
    train_wmse_vae[:,:,ii] = \
        np.array([rolling_mse(train_data[x,:,0], train_reconstruction_vae[x,:,ii], window) \
            for x in np.arange(n_train)])
    print('Completed: ' + str(ii))

# ---------------------------------------------------------------------------
## window mse threshold

train_wmse_pca_mean = np.mean(train_wmse_pca, axis = 1)
train_wmse_vae_mean = np.mean(train_wmse_vae, axis = 1)

threshold_wmse_pca = np.percentile(train_wmse_pca_mean, threshold_percentile, axis = 0)
threshold_wmse_vae = np.percentile(train_wmse_vae_mean, threshold_percentile, axis = 0)

# ---------------------------------------------------------------------------
## window mse threshold

test_wmse_label_pca = np.zeros_like(test_wmse_pca)
test_wmse_label_vae = np.zeros_like(test_wmse_vae)

def window_mse_label(window_mse, threshold):
    label = np.array([[np.any((window_mse > threshold)[y, \
        np.max((0, x - half_window)):np.min((x + half_window, sample_length))]) \
        for x in np.arange(sample_length)] for y in np.arange(n_test)])  
    return label

for ii in np.arange(n_dims):
    test_wmse_label_pca[:,:,ii] = \
        window_mse_label(test_wmse_pca[:,:,ii], threshold_wmse_pca[ii])
    test_wmse_label_vae[:,:,ii] = \
        window_mse_label(test_wmse_vae[:,:,ii], threshold_wmse_vae[ii])

# ---------------------------------------------------------------------------
## window mse threshold

# CHANGE THIS AS REQUIRED TO DISPLAY ARTEFACT DETECTION vs ANNOTATION FOR 
# EACH LATENT DIMENSION
latent_dim = 5
jj = np.where(latent_dim == np.array(latent_dims))[0]

with PdfPages('display_vae_ws_ld' + np.str(latent_dim) + '.pdf') as pdf:

    fig = plt.figure(figsize = (20, 25))

    colors = plt.get_cmap('tab10').colors

    for ii in np.arange(n_test):
        ll = ii % 12

        ax = plt.subplot(4, 3, ll + 1)
        plt.plot(t, test_data[ii], color = (0, 0, 0))
        plt.plot(t, test_reconstruction_vae[ii,:,jj][0, :], color = colors[1])
        plot_poly(t, test_data[ii], test_label_ws[ii], col = colors[0])
        plot_poly(t, test_data[ii], test_wmse_label_vae[ii,:,jj], col = colors[1])

        ax.set_ylim([dmin, dmax])

        if ii % 12 == 11:
            pdf.savefig()
            plt.close()

            fig = plt.figure(figsize = (20, 25))

    pdf.savefig()
    plt.close()

latent_dim = 20

with PdfPages('display_vae_s_ld' + np.str(latent_dim) + '.pdf') as pdf:

    fig = plt.figure(figsize = (20, 25))

    colors = plt.get_cmap('tab10').colors

    for ii in np.arange(n_test):
        ll = ii % 12

        plt.subplot(4, 3, ll + 1)
        plt.plot(t, bad_data[ii], color = (0, 0, 0))
        plt.plot(t, bad_prediction_vae[ii,:,jj][0, :], color = colors[1])

        if test_label_s[ii] == 1:
            verts = np.zeros((1, 4, 2))
            ts = np.array([t[0], t[-1], t[-1], t[0]])
            ys = np.array([dmin, dmin, dmax, dmax])
            verts[0] = list(zip(ts, ys))
            from matplotlib.collections import PolyCollection
            poly = PolyCollection(verts, facecolors = colors[0], \
                edgecolors = None)
            poly.set_alpha(0.2)
            ax.add_collection(poly)

        if test_mse_vae[ii, jj] > threshold_mse_vae[jj]:
            verts = np.zeros((1, 4, 2))
            ts = np.array([t[0], t[-1], t[-1], t[0]])
            ys = np.array([dmin, dmin, dmax, dmax])
            verts[0] = list(zip(ts, ys))
            from matplotlib.collections import PolyCollection
            poly = PolyCollection(verts, facecolors = colors[1], \
                edgecolors = None)
            poly.set_alpha(0.2)
            ax.add_collection(poly)

        ax.set_ylim([dmin, dmax])

        if ii % 12 == 11:
            pdf.savefig()
            plt.close()

            fig = plt.figure(figsize = (20, 25))

    pdf.savefig()
    plt.close()

# ---------------------------------------------------------------------------
## proportion correct (within sample)

pcorrect_pca = np.sum(np.equal(test_label_ws, test_wmse_label_pca), axis = 1)
pcorrect_pca /= sample_length
pcorrect_a_pca = np.sum(np.logical_and(test_label_ws, test_wmse_label_pca), axis = 1)
pcorrect_a_pca /= np.sum(test_wmse_label_pca, axis = 1) * pcorrect_a_pca / pcorrect_a_pca
pcorrect_n_pca = np.sum(np.logical_and(test_label_ws == 0, test_wmse_label_pca == 0), axis = 1)
pcorrect_n_pca /= np.sum(test_wmse_label_pca == 0, axis = 1) * pcorrect_n_pca / pcorrect_n_pca

pcorrect_vae = np.sum(np.equal(test_label_ws, test_wmse_label_vae), axis = 1)
pcorrect_vae /= sample_length
pcorrect_a_vae = np.sum(np.logical_and(test_label_ws, test_wmse_label_vae), axis = 1)
pcorrect_a_vae /= np.sum(test_wmse_label_vae, axis = 1) * pcorrect_a_vae / pcorrect_a_vae
pcorrect_n_vae = np.sum(np.logical_and(test_label_ws == 0, test_wmse_label_vae == 0), axis = 1)
pcorrect_n_vae /= np.sum(test_wmse_label_vae == 0, axis = 1) * pcorrect_n_vae / pcorrect_n_vae

# ---------------------------------------------------------------------------
## within sample performance

def zero_one_nan(data):
    data = np.array(data)
    data[data == 0] = np.nan
    data[data == 1] = np.nan
    return data

with PdfPages('within_sample.pdf') as pdf:

    fig = plt.figure(figsize = (8.5, 5))
    gs = gridspec.GridSpec(1, 2)

    colors = plt.get_cmap('Set1').colors

    ax_pca = plt.subplot(gs[0])
    for ii in np.arange(n_dims):
        plt.scatter(ii - 0.25 + np.random.random(size = n_test) * 0.1, \
            zero_one_nan(pcorrect_pca[:,ii]), color = colors[3], s = 0.1)
        plt.scatter(ii - 0.2, 1, color = colors[3], s = 0.1 * np.sum(pcorrect_pca[:,ii] == 1))
        plt.scatter(ii - 0.2, 0, color = colors[3], s = 0.1 * np.sum(pcorrect_pca[:,ii] == 0))
        plt.scatter(ii - 0.3, np.mean(pcorrect_pca, axis = 0)[ii], \
            color = colors[3], s = 15, marker = 's')

        plt.scatter(ii + np.random.random(size = n_test) * 0.1, \
            zero_one_nan(pcorrect_a_pca[:,ii]), color = colors[6], s = 0.1)
        plt.scatter(ii + 0.05, 1, color = colors[6], s = 0.1 * np.sum(pcorrect_a_pca[:,ii] == 1))
        plt.scatter(ii + 0.05, 0, color = colors[6], s = 0.1 * np.sum(pcorrect_a_pca[:,ii] == 0))
        plt.scatter(ii - 0.05, np.nanmean(pcorrect_a_pca, axis = 0)[ii], \
            color = colors[6], s = 15, marker = 's')

        plt.scatter(ii + 0.25 + np.random.random(size = n_test) * 0.1, \
            zero_one_nan(pcorrect_n_pca[:,ii]), color = colors[4], s = 0.1)
        plt.scatter(ii + 0.3, 1, color = colors[4], s = 0.1 * np.sum(pcorrect_n_pca[:,ii] == 1))
        plt.scatter(ii + 0.3, 0, color = colors[4], s = 0.1 * np.sum(pcorrect_n_pca[:,ii] == 0))
        plt.scatter(ii + 0.2, np.nanmean(spe_pca, axis = 0)[ii], \
            color = colors[4], s = 15, marker = 's')

    ax_pca.set_xticks(np.arange(n_dims))
    ax_pca.set_xticklabels(latent_dims)
    ax_pca.set_yticks(np.arange(0, 1.1, 0.1))
    ax_pca.set_xlabel('Latent dimension')
    ax_pca.set_ylabel('% correctly identified')
    ax_pca.title('PCA')
    ax_pca.yaxis.grid()

    ax_vae = plt.subplot(gs[1], sharey = ax_pca)
    for ii in np.arange(n_dims):
        plt.scatter(ii - 0.25 + np.random.random(size = n_test) * 0.1, \
            zero_one_nan(pcorrect_vae[:,ii]), color = colors[3], s = 0.1)
        plt.scatter(ii - 0.2, 1, color = colors[3], s = 0.1 * np.sum(pcorrect_vae[:,ii] == 1))
        plt.scatter(ii - 0.2, 0, color = colors[3], s = 0.1 * np.sum(pcorrect_vae[:,ii] == 0))
        plt.scatter(ii - 0.3, np.mean(pcorrect_vae, axis = 0)[ii], \
            color = colors[3], s = 15, marker = 's')

        plt.scatter(ii + np.random.random(size = n_test) * 0.1, \
            zero_one_nan(pcorrect_a_vae[:,ii]), color = colors[6], s = 0.1)
        plt.scatter(ii + 0.05, 1, color = colors[6], s = 0.1 * np.sum(pcorrect_a_vae[:,ii] == 1))
        plt.scatter(ii + 0.05, 0, color = colors[6], s = 0.1 * np.sum(pcorrect_a_vae[:,ii] == 0))
        plt.scatter(ii - 0.05, np.nanmean(pcorrect_a_vae, axis = 0)[ii], \
            color = colors[6], s = 15, marker = 's')

        plt.scatter(ii + 0.25 + np.random.random(size = n_test) * 0.1, \
            zero_one_nan(pcorrect_n_vae[:,ii]), color = colors[4], s = 0.1)
        plt.scatter(ii + 0.3, 1, color = colors[4], s = 0.1 * np.sum(pcorrect_n_vae[:,ii] == 1))
        plt.scatter(ii + 0.3, 0, color = colors[4], s = 0.1 * np.sum(pcorrect_n_vae[:,ii] == 0))
        plt.scatter(ii + 0.2, np.nanmean(pcorrect_n_vae, axis = 0)[ii], \
            color = colors[4], s = 15, marker = 's')

    ax1.set_xticks(np.arange(n_latent_dims))
    ax1.set_xticklabels(latent_dims)
    ax1.set_xlabel('Latent dimension')
    ax1.title('VAE')
    ax1.yaxis.grid()
    plt.setp(ax1.get_yticklabels(), visible = False)

    pcorrect = mlines.Line2D([], [], lw = 0, marker = 'o', markersize = 3, \
        color = colors[3], label = 'Entire sample')
    pcorrect_a = mlines.Line2D([], [], lw = 0, marker = 'o', markersize = 3, \
        color = colors[6], label = 'Artefact within sample')
    pcorrect_n = mlines.Line2D([], [], lw = 0, marker = 'o', markersize = 3, \
        color = colors[4], label = 'Non-artefact within sample')
    blank = mlines.Line2D([], [], lw = 0, label = ' ')
    mean_marker = mlines.Line2D([], [], lw = 0, marker = 's', markersize = 5, \
        color = (0, 0, 0), label = 'Mean')

    legend = fig.legend(loc = 'lower center', bbox_to_anchor = (0.5, 0), \
        handles = [pcorrect, pcorrect_a, pcorrect_n, blank, mean_marker],
        title = '% correctly identified of:', ncol = 5)
    legend._legend_box.align = "left"

    gs.tight_layout(fig, rect = [0, 0.1, 1, 1])
    gs.update(wspace = 0, hspace = 0)

    pdf.savefig()
    plt.close()

# ---------------------------------------------------------------------------
## latent_embedding

# CHANGE THIS AS REQUIRED TO DISPLAY LATENT SPACE FOR 
# EACH LATENT DIMENSION
latent_dim = 5
jj = np.where(latent_dim == np.array(latent_dims))[0]

for vae_file in vae_files:
    with h5py.File(file_dir_vae + '/' + vae_file, 'r') as hf_vae:
        if 'latent_dim' in list(hf_vae.keys()):
            latent_dim_file = hf_vae['latent_dim'][()]
        else:
            try:
                suffix = int(vae_file.strip(file_prefix).strip('.h5'))
                latent_dim_file = read_var_slurm(suffix, 'latent_dim')
            except:
                pass
        if latent_dim_file == latent_dim:
            file_suffix = int(vae_file.strip(file_prefix).strip('.h5'))
            break

hf_vae = h5py.File(file_dir_vae + '/' + 'vae_results_' + file_suffix + '.h5', 'r')

z_train = hf_vae['z_train_prediction']
z_validation = hf_vae['z_validation_prediction']
z_test = hf_vae['z_test_prediction']

fig_dim = [2 * np.min((latent_dim, 4)), 2 * np.min((latent_dim, 4))] 
xymax = np.max((np.max(np.fabs(z_test[0,:,:])), np.max(np.fabs(z_test[0,:,:]))))
def bins_fun(binwidth):
    lim = (int(xymax / binwidth) + 1) * binwidth
    bins = np.arange(-lim, lim + binwidth, binwidth)
    return(bins)

with PdfPages('latent.pdf') as pdf:
  
    fig = plt.figure(figsize = fig_dim)
    gs = gridspec.GridSpec(2 * latent_dim + 3, 2 * latent_dim + 3)

    colors = plt.get_cmap('tab10').colors
    test_colors = [colors[x] for x in test_label_s.astype(int)]

    for ii in np.arange(latent_dim ** 2):
        row, column = np.divmod(ii, latent_dim)

        test_xmax = np.ceil(np.max(np.fabs(z_test[0,:,:])))

        if row == column:
            ax = plt.subplot(gs[row * 2:row * 2 + 2, column * 2 + 3:column * 2 + 5])
            plt.hist(z_train[0, :, row], bins = bins_fun(0.1), color = (0, 0, 0))
            ax.set_xlim([-5, 5])
            ax.set_xticks([-4, 0, 4])
            ax.set_yticks([0, 1000])
            ax.tick_params(length = 2)
            if row == 0:
                ax.set_xlabel('z$_' + np.str(row + 1) + '$', rotation = 'horizontal')
                ax.xaxis.set_label_position('top')
            if column == latent_dim - 1:
                ax.set_ylabel('    z$_' + np.str(row + 1) + '$', rotation = 'horizontal')
                ax.yaxis.set_label_position('right')

            ax = plt.subplot(gs[row * 2 + 3:row * 2 + 5, column * 2:column * 2 + 2])
            plt.hist([z_test[0, np.where(test_label_s == 0)[0], column], \
                z_test[0, np.where(test_label_s == 1)[0], column]], bins = bins_fun(1), \
                color = (colors[0], colors[1]), stacked = True)
            ax.set_xlim([-test_xmax, test_xmax])
            ax.set_ylim([0, 80])
            ax.set_xticks([])
            ax.set_yticks([50])
            ax.tick_params(length = 2)
            ax.set_xlabel('z$_' + np.str(row + 1) + '$', rotation = 'horizontal')
            ax.xaxis.set_label_position('top')
            ax.set_ylabel('    z$_' + np.str(row + 1) + '$', rotation = 'horizontal')
            ax.yaxis.set_label_position('right')
            if row == latent_dim - 1:
                ax.set_xticks([-10, 0, 10])
            if row == 0:
                ax.set_yticks([0, 50])

        if row < column:
            ax = plt.subplot(gs[row * 2:row * 2 + 2, column * 2 + 3:column * 2 + 5])
            plt.scatter(z_train[0, :, row], z_train[0, :, column], s = 0.1, \
                color = (0, 0, 0), lw = 0)
            ax.set_xlim([-5, 5])
            ax.set_ylim([-5, 5])
            ax.set_xticks([-4, 0, 4])
            ax.set_yticks([-4, 0, 4])
            ax.tick_params(length = 2, pad = 2)
            ax.tick_params(direction = 'in', axis = 'x')
            ax.tick_params(labelright = False, labeltop = True, axis = 'y')
            if row + 1 != column:
                plt.setp(ax.get_yticklabels(), visible=False)
            if row == 0:
                ax.set_xlabel('z$_' + np.str(column + 1) + '$', rotation = 'horizontal')
                ax.xaxis.set_label_position('top')
            if column == latent_dim - 1:
                ax.set_ylabel('    z$_' + np.str(row + 1) + '$', rotation = 'horizontal')
                ax.yaxis.set_label_position('right')

        if row > column:
            ax = plt.subplot(gs[row * 2 + 3:row * 2 + 5, column * 2:column * 2 + 2])
            plt.scatter(z_train[0, :, column], z_train[0, :, row], s = 0.3, \
                color = (0, 0, 0), lw = 0)
            plt.scatter(z_test[0, :, column], z_test[0, :, row], s = 2, \
                color = test_colors, lw = 0)
            ax.set_xlim([-test_xmax, test_xmax])
            ax.set_ylim([-test_xmax, test_xmax])
            ax.set_xticks([-10, 0, 10])
            ax.set_yticks([-10, 0, 10])
            ax.tick_params(length = 2)
            if row != latent_dim - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
                ax.tick_params(direction = 'in', axis = 'x')
            if column != 0:
                plt.setp(ax.get_yticklabels(), visible=False)

    train = mlines.Line2D([], [], lw = 0, marker = 'o', markersize = 4, \
        color = (0, 0, 0), label = 'Training')
    test_artefact = mlines.Line2D([], [], lw = 0, marker = 'o', markersize = 4, \
        color = colors[1], label = 'Test (artefact)')
    test_none = mlines.Line2D([], [], lw = 0, marker = 'o', markersize = 4, \
        color = colors[0], label = 'Test (no artefact)')

    legend = fig.legend(loc = 'lower center', bbox_to_anchor = (0.5, 0), \
        handles = [train, test_artefact, test_none], ncol = 3)

    gs.tight_layout(fig, rect = [0, 0.05, 1, 1])
    gs.update(wspace = 0, hspace = 0)

    pdf.savefig()
    plt.close()

# ---------------------------------------------------------------------------
## example test reconstruction

# CHANGE THIS AS REQUIRED TO DISPLAY EXAMPLE TEST RECONSTRUCTIONS FOR 
# EACH LATENT DIMENSION
latent_dim = 5
jj = np.where(latent_dim == np.array(latent_dims))[0]

# CHANGE THIS AS REQUIRED TO CHOOSE A SUBSET OF THE TEST SET, LENGTH 6
selected_idx = [2, 5, 4, 22, 14, 48]

with PdfPages('example_test.pdf') as pdf:

    fig = plt.figure(figsize = (10, 5))
    gs = gridspec.GridSpec(2, 3)

    colors = plt.get_cmap('tab10').colors

    ymax = 12
    ymin = -8

    for ii in np.arange(6):
        if ii == 0:
            ax0 = plt.subplot(gs[ii])
        else:
            ax = plt.subplot(gs[ii], sharex = ax0, sharey = ax0)

        ll = selected_idx[ii]

        test_label_stacked = np.hstack(test_label_ws[ll])

        plot_poly(t, test_data[ll], \
            np.logical_and(test_label_ws[ll,:,0], np.logical_not(test_wmse_label_vae[ll,:,jj])), \
            col = colors[3], alpha = 0.6)
        plot_poly(t, test_data[ll], \
            np.logical_and(np.logical_not(test_label_ws[ll,:,0]), test_wmse_label_vae[ll,:,jj]), \
            col = colors[1], alpha = 0.2)
        plot_poly(t, test_data[ll], \
            np.logical_and(test_label_ws[ll,:,0], test_wmse_label_vae[ll,:,jj]), \
            col = .colors[2], alpha = 0.4)

        plt.plot(t, test_data[ll], color = (0, 0, 0), label = 'Data')
        plt.plot(t, test_reconstruction_vae[ll,:,jj][0, :], \
            color = colors[1], label = 'Reconstruction')

        plt.gca().set_yticks(np.arange(-8, 12, 4))

        plt.setp(plt.gca().get_xticklabels(), visible = False)
        plt.setp(plt.gca().get_yticklabels(), visible = False)

        if ii % 3 == 0:
            plt.setp(plt.gca().get_yticklabels(), visible = True)   
            plt.gca().set_ylabel('ABP (mmMg)')
        if ii // 3 != 0:
            plt.setp(plt.gca().get_xticklabels(), visible = True)
            plt.gca().set_xlabel('Time (s)')

        plt.gca().set_ylim([ymin, ymax])

    mdata = mlines.Line2D([], [], color = (0, 0, 0), label = 'Data')
    mreconstruction = mlines.Line2D([], [], color = colors[1], label = 'Reconstruction')
    annotated = mlines.Line2D([], [], lw = 0, marker = 's', markersize = 5, \
        color = colors[3], alpha = 0.6, label = 'Not identified')
    correct = mlines.Line2D([], [], lw = 0, marker = 's', markersize = 5, \
        color = colors[2], alpha = 0.4, label = 'Correctly identified')
    incorrect = mlines.Line2D([], [], lw = 0, marker = 's', markersize = 5, \
        color = colors[1], alpha = 0.2, label = 'Incorrectly identified')

    fig.legend(loc = 'lower center', bbox_to_anchor = (0.5, 0), ncol = 5,
        handles = [mdata, mreconstruction, annotated, correct, incorrect])

    gs.tight_layout(fig, rect = [0, 0.05, 1, 1])
    gs.update(wspace = 0, hspace = 0)

    pdf.savefig()
    plt.close()






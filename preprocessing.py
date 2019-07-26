## VARATIONAL AUTOENCODER FOR INTENSIVE CARE UNIT PHYSIOLOGICAL DATA

# ---------------------------------------------------------------------------
## modules

import numpy as np
import matplotlib.pyplot as plt
import h5py

import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
from matplotlib.backends.backend_pdf import PdfPages

from functions import load_h5, ReturnValue
from functions import plot_index, plot_block_section, index_to_time
from functions import dtype_change, preprocess_index, preprocess_quality
from functions import range_label, local_label, quality_label
from functions import join_label, add_label

# ---------------------------------------------------------------------------
## ABP waveform data - preprocessing

# CHANGE AS REQUIRED
file_path = '/Users'
data_dir = 'data'
file_name = 'example_file.hdf5'

hf = load_h5(file_path + '/' + data_dir + '/' + file_name, checks = False)

data = hf.data['waves/abp'][()]
index = hf.data['waves/abp.index'][()]
index = preprocess_index(index)

quality = hf.data['waves/abp.quality'][()]
qualityRef = hf.data['definitions/qualityRef'][()]
starttime = hf.data.attrs['dataStartTimeUnix'].astype(int)
quality = preprocess_quality(quality, qualityRef, starttime)

label_r = range_label(data, data_min = -5, data_max = 240, label = 1)
label_l = local_label(data, index, tol_min = 0.5, tol_max = 80, t = 0.25)
label_q = quality_label(quality, index)

label_r = join_label(label_r, index)
label_l = join_label(label_l, index)
label_q = join_label(label_q, index)

label_l['label'] = label_l['label'] + np.size(np.unique(label_r['label']))
label_q['label'] = label_q['label'] \
    + np.size(np.unique(label_r['label'])) + np.size(np.unique(label_l['label']))

label_l = label_l[list(label_r.dtype.names)]
label_q = label_q[list(label_r.dtype.names)]

label_merge = add_label((label_r, label_l, label_q), index)

# ---------------------------------------------------------------------------
## sampling for samples

t_interval = 10
# ASSUMPTION THAT RECORDING FREQUENCY IS CONSTANT FOR ALL
# SECTIONS OF THE DATA
# IF NOT, SOME STANDARDISATION IN TIME NEEDS TO OCCUR
sample_length = np.ceil(t_interval * index['frequency'][0]).astype(int)
t = np.arange(sample_length) * t_interval / sample_length

# ---------------------------------------------------------------------------
## sampling for test set

t_interval_sampling = 100

sampling_length = np.ceil(t_interval_sampling * index['frequency'][0]).astype(int)

label = label_merge[label_merge['label'] == 0]
startidx = label['startidx']
length = label['length']

max_x = np.sum(np.floor(index['length'] / sampling_length)).astype(int)

p = np.zeros((max_x, ))
idx = np.zeros((max_x, )).astype(int)

idx_current = 0
kk = 0
jj = 0

for x in np.arange(max_x):
    if idx_current + sampling_length > index['startidx'][jj] + index['length'][jj]:
        jj += 1
        idx_current = index['startidx'][jj]

    idx[x] = idx_current

    if idx_current + sampling_length < startidx[kk] + length[kk]:
        p[x] = 1
    else:
        p[x] = startidx[kk] + length[kk] - idx_current
        kk += 1
        while idx_current + sampling_length > startidx[kk] + length[kk]:
            p[x] += length[kk]
            kk += 1
        kk -= 1
        p[x] += idx_current + sampling_length - startidx[kk] - length[kk]
        p[x] /= sampling_length
        kk += 1

    idx_current += sampling_length

p = 1 - p
p_copy = p.copy()
p /= np.sum(p)

n_test = 200

test_idx = np.random.choice(idx, n_test, p = p) \
    + np.random.randint(sampling_length, size = n_test)

# ---------------------------------------------------------------------------
## preprocessing

# CHANGE AS REQUIRED
starttime = 86290
endtime = 86390

with PdfPages('preprocessing.pdf') as pdf:

    fig = plt.figure(figsize=(11, 4))

    gs = gridspec.GridSpec(3, 10)
    gs.update(wspace = 0, hspace = 0.001)

    colors = plt.get_cmap('tab10').colors
    colors = np.vstack(((0, 0, 0), [colors[x] \
        for x in np.arange(np.size(np.unique(label_merge['label'])))]))

    # subfigure (a)
    ax_a = plt.subplot(gs[0, :6])
    ax_a.set_xlim([hf.data.attrs['dataStartTimeUnix'].astype(int), \
        hf.data.attrs['dataEndTimeUnix'].astype(int)])
    plot_index(index, label = label_merge, ax = ax_a, y = 1, cmap = colors)

    ax_a.set_yticks([])
    ax_a.set_xlabel('Time (s)')

    # subfigure (b)
    ax_b = plt.subplot(gs[0, 7:])
    plot_index(index, label = label_merge, ax = ax_b, cmap = colors)

    starttime_0 = starttime + index['starttime'][0]
    endtime_0 = endtime + index['starttime'][0]

    ax_b.set_xlim([starttime_0, endtime_0])
    ax_b.set_yticks([])
    ax_b.set_xlabel('Time (s)')

    # subfigure (c)
    ax_c = plt.subplot(gs[1:, :6], sharex = ax_a)
    time_p = [index_to_time(index, idx[x]) for x in np.arange(np.size(idx))]
    plt.plot(time_p, p_copy)

    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)

    xticks = ax_c.get_xticks()
    ax_c.set_xticks(np.arange(index['starttime'][0], xticks[-1], 86400))
    ax_c.set_xticklabels(np.arange(0, xticks[-1] - index['starttime'][0], 86400).astype(int))

    yticks = ax_c.get_yticks()
    ax_c.set_yticks(yticks[1:-1])

    xlim = ax_c.get_xlim()
    ax_c.set_xlim([xlim[0] * 0.99, xlim[1] * 1.01])
    ylim = ax_c.get_ylim()
    ax_c.set_ylim([ylim[0], np.min((1, ylim[1] * 1.2))])

    ax_c.set_xlabel('Time (s)')
    ax_c.set_ylabel('Proportion marked')

    # subfigure (d)
    ax_d = plt.subplot(gs[1:, 7:], sharex = ax_b)

    block = np.max((np.argmin(index['starttime'] < starttime_0) - 1, 0))
    plot_block_section(data, index[block], label_merge, starttime_0, endtime_0)

    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)

    xticks = ax_d.get_xticks()
    ax_d.set_xticks(np.round(xticks - index['starttime'][block], -1) + index['starttime'][block])
    ax_d.set_xticklabels(np.round(xticks - index['starttime'][block], -1).astype(int))

    yticks = ax_d.get_yticks()
    ax_d.set_yticks(yticks[1:-1])

    ax_d.set_xlim([starttime_0, endtime_0])
    ylim = ax_d.get_ylim()
    ax_d.set_ylim([ylim[0], ylim[1] * 1.3])

    ax_d.set_xlabel('Time (s)')
    ax_d.set_ylabel('ABP (mmHg)')

    # axis for legend
    ax_hidden = plt.subplot(gs[0, 6])
    plt.axis('off')

    nolab = mlines.Line2D([], [], color = (0, 0, 0), label = 'None')
    lab1 = mlines.Line2D([], [], color = colors[1], label = '(i)')
    lab2 = mlines.Line2D([], [], color = colors[2], label = '(ii)')
    lab3 = mlines.Line2D([], [], color = colors[3], label = '(iii)')
    lab4 = mlines.Line2D([], [], color = colors[5], label = '(iv)')
    plt.legend(loc = 'center', \
        handles = [nolab, lab1, lab2, lab3, lab4], bbox_to_anchor = (0, 0.4))

    # label subfigures
    plt.figtext(0.02, 0.95, '(a)')
    plt.figtext(0.7, 0.95, '(b)')
    plt.figtext(0.02, 0.7, '(c)')
    plt.figtext(0.7, 0.7, '(d)')

    # layout
    gs.tight_layout(fig)

    pdf.savefig()
    plt.close()

# ---------------------------------------------------------------------------
## add labels to test set samples

label_test = np.array([tuple((np.sort(test_idx)[x], sample_length, 1)) \
    for x in np.arange(np.size(test_idx))],
        dtype = [('startidx', 'i8'), ('length', 'i8'), ('label', 'i4')])

label_test['label'] = label_test['label'] + np.size(np.unique(label_merge['label'])) - 1

label_test = label_test[list(label_r.dtype.names)]

label_final = add_label((label_r, label_l, label_q, label_test), index)

# ---------------------------------------------------------------------------
## 

def get_chunks(data, index, t_interval, **kwargs):
    '''
    Create chunks of data of length t_interval (in s).

    Parameters
    ----------
    data: array
    index: array
    t_interval: float
        Time t_interval: break data into chunks of this length.
        Default is 10s.

    Keyword arguments
    ----------
    label: array
    '''

    # assuming data frequency is constant across all channel
    # if not then need to encode timestep as well as data value
    #   the output

    # if not np.all(index['frequency'] == index['frequency'][0]):
    #     print('Assumption that data frequency is constant across all channel')
    #     return

    if 'label' in kwargs.keys():
        label = kwargs['label']
        label = label[label['label'] == 0]
        length = label['length']
        startidx = label['startidx']
    else:
        length = index['length']
        startidx = index['startidx']

    sample_length = np.ceil(t_interval * index['frequency'][0]).astype(int)
    max_samples = np.sum(np.floor(length / sample_length)).astype(int)
    data_array = np.zeros((max_samples, sample_length))

    idx_array = np.zeros((max_samples, )).astype(int)

    idx = 0
    jj = 0
    for x in np.arange(max_samples):
        if idx + sample_length > startidx[jj] + length[jj]:
            jj = np.argmax(np.all((length > sample_length, startidx > idx), axis = 0))
            idx = startidx[jj]
        idx_array[x] = idx
        data_array[x, :] = data[idx:idx + sample_length]
        idx += sample_length

    return data_array

data_array = get_chunks(data, index, 10, label = label_final)

def preprocess(data_array):
    '''
    Shuffle and split data samples into test and validation, then 
    standardise by the training set.

    Parameters
    ----------
    data_array: array
    '''

    n_samples = np.shape(data_array)[0]

    shuffle = np.arange(n_samples)

    np.random.shuffle(shuffle)

    data_array = data_array[shuffle]

    split = np.array([0.9, 0.1])
    idx = np.ceil(n_samples * np.cumsum(split)).astype(int)

    data_array = data_array.reshape(data_array.shape + (1, ))

    train_data = data_array[:idx[0], :, :]
    validation_data = data_array[idx[0]:, :, :]

    mean = np.mean(train_data)
    train_data -= mean
    std = np.std(train_data)
    train_data /= std

    validation_data -= mean
    validation_data /= std

    return (train_data, validation_data, mean, std, shuffle)

(train_data, validation_data, mean, std, shuffle) = preprocess(data_array)

n_train = np.shape(train_data)[0]

# ---------------------------------------------------------------------------
## test set

test_data = np.zeros((n_test, sample_length, 1))

for ii in np.arange(n_test):
    test_data[ii, :, 0] = data[test_idx[ii]:test_idx[ii] + sample_length]

test_data -= mean
test_data /= std

dmin = np.min(test_data)
dmax = np.max(test_data)

# ---------------------------------------------------------------------------
## save

file_name_input = 'data_input.h5'

hf = h5py.File(file_path + '/' + file_name_input, 'w')

hf.create_dataset('train_data', data = train_data)
hf.create_dataset('validation_data', data = validation_data)
hf.create_dataset('test_data', data = test_data)
hf.create_dataset('shuffle', data = shuffle)
hf.create_dataset('mean', data = mean)
hf.create_dataset('std', data = std)
hf.create_dataset('test_idx', data = test_idx)

hf.close()

# ---------------------------------------------------------------------------
## display 

with PdfPages('display_test.pdf') as pdf:

    fig = plt.figure(figsize = (20, 25))

    for ii in np.arange(n_test):
        ll = ii % 12

        ax = plt.subplot(4, 3, ll + 1)
        plt.plot(t, test_data[ii], color = (0, 0, 0), label = 'Input')

        ax.set_ylim([dmin, dmax])

        if ii % 12 == 11:
            pdf.savefig()
            plt.close()

            fig = plt.figure(figsize = (20, 25))

    pdf.savefig()
    plt.close()



## VARATIONAL AUTOENCODER FOR INTENSIVE CARE UNIT PHYSIOLOGICAL DATA

import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

from scipy.stats import norm

import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.lines as mlines

# ---------------------------------------------------------------------------

try:
    file_suffix = sys.argv[1]
except:
    print('Must include file number as argument')
    sys.exit()

if ('--input_dir' in sys.argv):
    try:
        input_dir = sys.argv[sys.argv.index('--input_dir') + 1]
        input_dir = str(input_dir) + '/'
    except:
        input_dir = './'

hf_data = h5py.File('data_array.h5', 'r')
hf_results = h5py.File(input_dir + 'vae_results_' + file_suffix + '.h5', 'r')

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

if 'plots' not in get_immediate_subdirectories('.'):
    os.mkdir('plots')

output_dir = 'plots/'

if ('--output_dir' in sys.argv):
    try:
        if os.path.isdir(sys.argv[sys.argv.index('--output_dir') + 1]):
            output_dir = sys.argv[sys.argv.index('--output_dir') + 1]
            if output_dir[-1] != '/':
                output_dir = output_dir + '/'
        else:
            print('--output_dir not a valid directory')
    except:
        pass

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

t_interval = 10
sample_length = np.shape(hf_data['train_data'])[1]
t = np.arange(sample_length) * t_interval / sample_length

# ---------------------------------------------------------------------------

if ('--reconstructions' in sys.argv):

    n_pages = 10
    n_columns = 4
    n_plots = n_pages * n_columns

    data = np.zeros((n_plots, sample_length, 6))

    n_train_plots = np.min((n_plots, np.shape(hf_data['train_data'])[0]))
    n_val_plots = np.min((n_plots, np.shape(hf_data['validation_data'])[0]))
    n_test_plots = np.min((n_plots, np.shape(hf_data['test_data'])[0]))

    data[:n_train_plots,:,0] = hf_data['train_data'][:n_train_plots,:,0] 
    data[:n_train_plots,:,1] = hf_results['train_prediction'][:n_train_plots,:,0]
    data[:n_val_plots,:,2] = hf_data['validation_data'][:n_val_plots,:,0] 
    data[:n_val_plots,:,3] = hf_results['validation_prediction'][:n_val_plots,:,0]
    data[:n_test_plots,:,4] = hf_data['test_data'][:n_test_plots,:,0] 
    data[:n_test_plots,:,5] = hf_results['test_prediction'][:n_test_plots,:,0] 

    dmin = np.floor(np.max((np.min(data), -8))) - 1
    dmax = np.ceil(np.min((np.max(data), 15))) + 1

    with PdfPages(output_dir + 'reconstructions_' + file_suffix + '.pdf') as pdf:

        fig = plt.figure(figsize = (20, 25))
        gs = gridspec.GridSpec(n_columns, 3)

        colors = plt.get_cmap('tab10').colors
        colors = np.vstack(colors)[:3]
        colors = np.insert(colors, np.arange(3), (0, 0, 0), axis = 0)

        for ii in np.arange(n_plots * 3):
            jj = ii % 3
            kk = ii // 3
            ll = ii % (3 * n_columns)

            if jj == 0:
                plot_ind = kk < n_train_plots
            elif jj == 1:
                plot_ind = kk < n_val_plots
            elif jj == 2:
                plot_ind = kk < n_test_plots

            if plot_ind == 1:
                ax = plt.subplot(gs[ll])
                plt.plot(t, data[kk,:, 2 * jj], color = colors[2 * jj], \
                    label = 'Input')
                plt.plot(t, data[kk,:, 2 * jj + 1], color = colors[2 * jj + 1], \
                    label = 'Output')

                ax.set_ylim([dmin, dmax])

            if kk == 0:
                if jj == 0:
                    plt.title('Training')
                    plt.legend(loc = 1)
                elif jj == 1:
                    plt.title('Validation')
                    plt.legend(loc = 1)
                elif jj == 2:
                    plt.title('Test')
                    plt.legend(loc = 1)

            if jj != 0:
                plt.setp(ax.get_yticklabels(), visible = False)
            if kk != 3:
                plt.setp(ax.get_xticklabels(), visible = False)

            if ii % (n_columns * 3) == (n_columns * 3 - 1) and kk != n_pages:
                gs.tight_layout(fig)
                gs.update(wspace = 0, hspace = 0)

                pdf.savefig()
                plt.close()

                fig = plt.figure(figsize = (20, 25))
                gs = gridspec.GridSpec(n_columns, 3)

        gs.tight_layout(fig)
        gs.update(wspace = 0, hspace = 0)

        pdf.savefig()
        plt.close()

# ---------------------------------------------------------------------------

if ('--reconstructions_test' in sys.argv):

    n_pages = 10
    n_columns = 4
    n_rows = 3
    n_plots = n_pages * n_columns * n_rows

    n_test_plots = np.max((n_plots, np.shape(hf_data['test_data'])[0]))

    data = np.zeros((n_test_plots, sample_length, 2))
    data[:, :, 0] = hf_data['test_data'][:n_test_plots, :, 0] 
    # data[:, :, 1] = hf_results['test_prediction'][:n_test_plots,:,0]
    data[:, :, 1] = hf_results['test_prediction'][0, :n_test_plots, :]

    dmin = np.floor(np.max((np.min(data), -8))) - 1
    dmax = np.ceil(np.min((np.max(data), 15))) + 1

    with PdfPages(output_dir + 'reconstructions_test_' + file_suffix + '.pdf') as pdf:

        fig = plt.figure(figsize = (20, 25))
        gs = gridspec.GridSpec(n_columns, n_rows)

        colors = plt.get_cmap('tab10').colors

        for ii in np.arange(n_test_plots):
            jj = ii % n_rows
            kk = ii // n_rows
            ll = ii % (n_columns * n_rows)

            ax = plt.subplot(gs[ll])
            plt.plot(t, data[ii, :, 0], color = (0, 0, 0), label = 'Input')
            plt.plot(t, data[ii, :, 1], color = colors[2], label = 'Output')

            if ii == 0:
                plt.legend(loc = 1)

            ax.set_ylim([dmin, dmax])

            if jj != 0:
                plt.setp(ax.get_yticklabels(), visible = False)
            if kk != n_rows:
                plt.setp(ax.get_xticklabels(), visible = False)

            if ll == (n_columns * n_rows - 1) and kk != n_pages:
                gs.tight_layout(fig)
                gs.update(wspace = 0, hspace = 0)

                pdf.savefig()
                plt.close()

                fig = plt.figure(figsize = (20, 25))
                gs = gridspec.GridSpec(n_columns, n_rows)

        gs.tight_layout(fig)
        gs.update(wspace = 0, hspace = 0)

        pdf.savefig()
        plt.close()

# ---------------------------------------------------------------------------

if ('--latent' in sys.argv):

    z_train = hf_results['z_train_prediction']
    z_test = hf_results['z_test_prediction']

    if 'latent_dim' in hf_results.keys():
        latent_dim = hf_results['latent_dim'][()]
    else:
        latent_dim = np.shape(hf_results['z_train_prediction'])[2]

    xymax = np.max((np.max(np.fabs(z_test[0,:,:])), np.max(np.fabs(z_test[0,:,:]))))
    def bins_fun(binwidth):
        lim = (int(xymax / binwidth) + 1) * binwidth
        bins = np.arange(-lim, lim + binwidth, binwidth)
        return(bins)

    with PdfPages(output_dir + 'latent_' + file_suffix + '.pdf') as pdf:
      
        fig = plt.figure(figsize = (8, 8))
        gs = gridspec.GridSpec(2 * latent_dim + 3, 2 * latent_dim + 3)

        colors = plt.get_cmap('tab10').colors

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
                plt.hist(z_test[0, :, column], bins = bins_fun(1), color = colors[0])
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
                    color = colors[0], lw = 0)
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

if ('--embedding' in sys.argv):

    if 'latent_dim' in hf_results.keys():
        latent_dim = hf_results['latent_dim'][()]
    else:
        latent_dim = np.shape(hf_results['z_prediction'])[2]

    z_grid = hf_results['z_grid']
    z_embedding = hf_results['z_embedding']

    n_grid = np.argmax(z_grid[1:, 0] == z_grid[0, 0]) + 1
    n_grid = 5
    n_plots = np.shape(z_grid)[0]
    n_pages = int(n_plots / n_grid ** 2)

    dmin = np.floor(np.min(z_embedding)) - 1
    dmax = np.ceil(np.max(z_embedding)) + 1

    def to_idx(n, n_grid, latent_dim):
        return (n // n_grid ** np.arange(latent_dim)) % n_grid

    fig_dim = (n_grid ** 2 / 4 * 1.5, n_grid ** 2 / 4)

    grid = norm.ppf(np.linspace(0.05, 0.95, n_grid))

    idx = np.array([np.flip(to_idx(x, latent_dim, 2), axis = 0) \
        for x in np.arange(latent_dim * (latent_dim - 1))])
    idx = np.array([x for x in idx if x[0] < x[1]])

    def title_fun(idx):
        title = ['z$_' + str(x) + '$' for x in idx]
        title = ' vs '.join(title)
        title_add = ['z$_' + str(x) + '$ = 0' for x in np.arange(latent_dim) if x not in idx]
        title_add = ', '.join(title_add)
        title = ', '.join([title, title_add])
        return title

    with PdfPages(output_dir + 'embedding_' + file_suffix + '.pdf') as pdf:

        fig = plt.figure(figsize = fig_dim)
        gs = gridspec.GridSpec(n_grid, n_grid)

        for ii in np.arange(n_plots):
            jj = ii % (n_grid ** 2)
            kk = ii // (n_grid ** 2)

            if jj == 0:
                plt.suptitle(title_fun(idx[kk]))

            ax = plt.subplot(gs[jj])
            plt.plot(z_embedding[ii], color = (0, 0, 0))

            ax.set_ylim([dmin, dmax])

            ax.set_xticks([])
            ax.set_yticks([])

            if jj % n_grid == 0:
                ax.set_ylabel('z$_' + np.str(idx[kk, 0]) + '$ = ' \
                    + np.str(np.round(grid[jj // n_grid], 2)))
            if jj // n_grid == n_grid - 1:
                ax.set_xlabel('z$_' + np.str(idx[kk, 1]) + '$ = ' \
                    + np.str(np.round(grid[jj % n_grid], 2)))

            if jj == n_grid ** 2 - 1 and jj > 0 and kk != n_pages - 1:
                gs.tight_layout(fig, rect = [0, 0.05, 1, 0.95])
                gs.update(wspace = 0, hspace = 0)

                pdf.savefig()
                plt.close()

                fig = plt.figure(figsize = fig_dim)
                gs = gridspec.GridSpec(n_grid, n_grid)

        gs.tight_layout(fig, rect = [0, 0.05, 1, 0.95])
        gs.update(wspace = 0, hspace = 0)

        pdf.savefig()
        plt.close()

# ---------------------------------------------------------------------------

if ('--loss' in sys.argv):

    with PdfPages(output_dir + 'loss_' + file_suffix + '.pdf') as pdf:
        fig = plt.figure()

        colors = plt.get_cmap('tab10').colors

        beta = hf_results['beta'][()]

        plt.plot(hf_results['kl_loss'] / beta, 'o', color = colors[1], label = 'KL (training)')
        plt.plot(hf_results['val_kl_loss'] / beta, color = colors[1], label = 'KL (validation)')

        plt.plot(hf_results['reconstruction_loss'], 'o', color = colors[2], \
            label = 'Reconstruction (training)')
        plt.plot(hf_results['val_reconstruction_loss'], color = colors[2], \
            label = 'Reconstruction (validation)')

        plt.plot(hf_results['loss'], 'o', color = colors[0], label = 'Loss (training)')
        plt.plot(hf_results['val_loss'], color = colors[0], label = 'Loss (validation)')

        plt.legend(loc = 1)

        ymax = np.max((np.median(hf_results['loss']) * 2, \
            np.median(hf_results['val_loss']) * 2))
        ymin = np.max((plt.gca().get_ylim()[0], 0))

        plt.gca().set_ylim([ymin, ymax])
        plt.grid()

        fig.tight_layout()

        pdf.savefig()
        plt.close()

# ---------------------------------------------------------------------------

hf_data.close()
hf_results.close()

# ---------------------------------------------------------------------------

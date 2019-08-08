# DeepClean: self-supervised artefact rejection for intensive care waveform data using generative deep learning

Usage:

preprocessing.py, functions.py require ICM+ hdf5 files with waveforms given as signals, though modifications can be made to these files such that similar datasets in hdf5 can be used. For more information about the structure of ICM+ hdf5 datasets, see https://www.ncbi.nlm.nih.gov/pubmed/29492546. It is not specifically designed to be run from the command line but from within python as required.

functions.py contains functions for reading, checking and marking ICM+ waveforms from hdf5 file.
preprocessing.py uses these functions for the preprocessing and displaying the ICM+ data as described within the manuscript, including splitting the data into training, validation and test sets. It is not specifically designed to be run from the command line but from within python as required.

analysis.py generates all the post-processing (such as MSE metric calculation and DeepClean performance) and figures within the manuscript. analysis.py requires several hdf5 files, containing data, artefact annotations and DeepClean VAE output files. It is not specifically designed to be run from the command line but from within python as required.

vae.py trains DeepClean using a hdf5 data file containing training, validation and test data, returning reconstructions, latent predictions and network loss history during training. Specific hyperparameters may be given at the command line when running the script, which will overwrite default hyperparameter choices, the code may be extended to similarly include further hyperparameters. Example usage is 'python vae.py --latent_dim 5'.

vae.sh is a bash script wrapper for GPU usage under slurm workload manager typically used in high performance computing. Hyperparameters given at the command line when running the script are passed through to vae.py. Example usage is 'slurm vae.sh --latent_dim 5'.

vae_plots.py returns a series of figures to visualise the output of vae.py. This includes reconstructions, latent space representations, reconstructions via the decoder from a latent grid embedding, and loss and validation loss history. Each required plot must be called at the command line when running the script and the first argument must give the file suffix relating to the DeepClean output file. Example usage is 'python vae_plots.py 1 --input_dir INPUT_DIR --output_dir OUTPUT_DIR --reconstructions --embedding --loss', where 1 is the file suffix.

vae_plots_dir.sh is a bash script wrapper for batch usage of vae_plots.py on an entire directory of DeepClean output files. Each required plot must be called at the command line when running the script and the first argument must give the input directory containing the DeepClean output files. Example usage is 'bash vae_plots_dir.sh INPUT_DIR --output_dir OUTPUT_DIR --reconstructions --embedding --loss'.


Abstract:

Waveform physiological data is important in the treatment of critically ill patients in the intensive care unit. Such recordings are susceptible to artefacts and interference, which must be removed before the data can be re-used for alerting or reprocessing for clinical or research purposes. The current gold-standard is human mark-up, which is painstaking and susceptible to reproducibility issues when such recordings may span many days.
In this work, we present DeepClean; a prototype artefact detection system using a one-dimensional convolutional, variational autoencoder deep neural network. As this technique is self-supervised and only requires easily-obtained `good' data for the training process, it holds a significant advantage in that it avoids any costly manual mark-up process. For the test case of invasive arterial blood pressure signal artefact detection, we demonstrate that our algorithm can detect the presence of an artefact within a 10-second sample of data with sensitivity and specificity around 90\%. Furthermore, the system was able to identify regions of artefact within such samples with high accuracy. We show that it significantly outperforms a baseline principle component analysis approach in both signal reconstruction and artefact detection. The model is generative and therefore may also be used for imputation of missing data. Accurate removal of artefacts reduces bias and uncertainty in clinical assessment of the patient and reduces the false negative rate of ICU alarms, and is therefore a key component in providing optimal clinical care.

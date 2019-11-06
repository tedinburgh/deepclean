$input_file = 'input.h5'
$output_file_suffix = None
$output_file = None
$output_logfile = True
$verbose = True
$save_weights = False
$load_weights = False

$latent_dim = 5
$Nz = 20

$batch_normalisation = False 
$dropout = 0.1 
$network = 'conv' 
$pool_size = 5 
$x_var = False

$train = True
$batch_size = 32
$epochs = 40
$patience = 8
$optimizer = 'RMSprop'
$learning_rate = 0.001
$kl_anneal = False
$temp_start
$temp_epochs

$min_z_var = 1e-5
$min_x_var = 0

$beta = 1

python vae_arg.py -i $input_file -output_file_suffix $output_file_suffix
-o $output_file -l $output_logfile -v $verbose -save_weights $save_weights
-load_weights $load_weights -ld $latent_dim -Nz $Nz

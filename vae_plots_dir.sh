#!/bin/bash
if [ "$1" == "-h" ]; then
  echo "Usage: `basename $0` Run the script vae_plots.py on every file in the directory 
  specified as the first argument. Subsequent arguments are passed to vae_plots.py."
  exit 0
fi

dir=$1

for hf_file in $dir/vae_results*
do 
    suffix=$(echo $hf_file | sed 's/.*\(vae_results.*h\).*/\1/' | tr -dc '0-9')
    python vae_plots.py $suffix --input_dir $dir ${@:2}
done

exit 0
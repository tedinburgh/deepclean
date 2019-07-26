#!/bin/bash

# Shell commands may precede the Python script.
echo "==============================================================="

cat vae.py

for jobno in $(ls -t slurm* | head -1 | tr -dc '0-9'); do python vae.py $@ --file_suffix $jobno; done;

echo "==============================================================="

exit 0

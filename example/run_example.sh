#!/bin/bash

# Run a mini example from start to end.

# First make a labelled file from header

# usage: run_example.sh <header input>. For this example: bash run_example.sh dense_mini.txt 

# Setup python environment. Change to your own

source /home/nurith/CS612/ProtTrans/bin/activate
output=$(basename ${1} .txt)

# make labelled file. Creates a labelled file according to heavy atom, 6 A distance.
#python fetch_and_label_pairs_matrix_heavy.py -o ${output}_out.txt --distance-mode heavy --threshold 6.0 --min-length 50 --max-length 300 --skip-existing $1

# The labelled file is called dense_mini_out.txt. The next step is remove redundancy. The output file is dense_mini_30.csv
python make_nonredundant_mmseqs.py -i ${output}_out.txt --threshold 0.3 --output ${output}_30.csv

# On a cluster, run sbatch run_example_chimera.sh


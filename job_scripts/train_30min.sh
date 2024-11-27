#!/bin/bash
#BSUB -q gpua100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process" 
#BSUB -J train_15min
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=10GB]"
#BSUB -W 00:03
#BSUB -o job_outputs/train_15min%J.out
#BSUB -e job_outputs/train_15min%J.err

source ~/venv/RecSys/bin/activate

python main.py \
    --experiment_name 'test_30min' \
    --num_epochs 5 \
    --dataset_name 'ebnerd_small'\

#!/bin/bash
#BSUB -q c02516
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process" 
#BSUB -J training
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=10GB]"
#BSUB -W 0:10
#BSUB -o job_outputs/training%J.out
#BSUB -e job_outputs/training%J.err

source ~/venv/RecSys/bin/activate

python Testing.py



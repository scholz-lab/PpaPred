#!/bin/bash
#SBATCH -o %j_predict.out
#SBATCH -e %j_predict.err
#SBATCH -D ./
#SBATCH -J predict
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=0-05:00:00
#SBATCH --mem=0

source activate sklearn-env2
python -u predict.py -in "/gpfs/soma_fs/scratch/src/boeger/data_gueniz/" -p ${1} -o "/gpfs/soma_fs/scratch/src/boeger/PpaPred_eren_35727184" -id $SLURM_JOB_ID
#python -u predict.py -in '/gpfs/soma_fs/gnb/gnb9201.bak/Mariannne Roca/MR_MS_pharaglowfiles/' -p ${1} -o "/gpfs/soma_fs/scratch/src/boeger/PpaPred_roca_35727184" -id $SLURM_JOB_ID
# python -u predict.py -in '/gpfs/soma_fs/scratch/src/hiramatsu' -p "240311_3" -o "/gpfs/soma_fs/scratch/src/boeger/PpaPred_hiramatsu_2403"
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
python -u predict.py -in "/gpfs/soma_fs/scratch/src/boeger/data_gueniz/ErenBoeger_2024" -p ${1} -o "/gpfs/soma_fs/scratch/src/boeger/PpaPred_eren_35727184/test_20240709" -id $SLURM_JOB_ID #-o "/gpfs/soma_fs/scratch/src/boeger/PpaPred_eren_35727184/ErenBoeger_2024"
#python -u predict.py -in '/gpfs/soma_fs/gnb/gnb9201.bak/Mariannne Roca/MR_MS_pharaglowfiles/' -p ${1} -o "/gpfs/soma_fs/scratch/src/boeger/PpaPred_roca_35727184" -id $SLURM_JOB_ID
# python -u predict.py -in '/gpfs/soma_fs/scratch/src/hiramatsu' -p "240311_3" -o "/gpfs/soma_fs/scratch/src/boeger/PpaPred_hiramatsu_2403"

mv $SLURM_JOB_ID"_predict.out" $SLURM_JOB_ID"_"${1}"_predict.out"
mv $SLURM_JOB_ID"_predict.err" $SLURM_JOB_ID"_"${1}"_predict.err"

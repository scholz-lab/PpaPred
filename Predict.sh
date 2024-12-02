#!/bin/bash
#SBATCH -o %j_predict.out
#SBATCH -e %j_predict.err
#SBATCH -D ./
#SBATCH -J predict
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=0-05:00:00
#SBATCH --mem=0

source activate sklearn-env2
python -u predict.py -p ${1} -id $SLURM_JOB_ID

mv $SLURM_JOB_ID"_predict.out" $SLURM_JOB_ID"_"${1}"_predict.out"
mv $SLURM_JOB_ID"_predict.err" $SLURM_JOB_ID"_"${1}"_predict.err"

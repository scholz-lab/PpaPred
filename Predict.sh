#!/bin/bash
#SBATCH -o %j_predict.out
#SBATCH -e %j_predict.err
#SBATCH -D ./
#SBATCH -J predict
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=1-00:00:00
#SBATCH --mem=0

#!/bin/bash
source activate sklearn-env2
python -u predict.py -in "/gpfs/soma_fs/scratch/src/boeger/data_gueniz/" -p "Exp1_WT_OP50" -o "/gpfs/soma_fs/scratch/src/boeger/PpaPred_eren_2403"

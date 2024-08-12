# PpaPred
### Prediction Pipeline for foraging behaviours of *Pristionchus pacificus* based on PharaGlow

#### Steps:
1. Download git repository:<br>
`git clone git@github.com:scholz-lab/PpaPred.git`

2. Install environmnet:<br>
`conda env create -f requirements.txt -n PpaPred`

3. Edit config and config_bath files, to specifiy path to your PharaGlow files, etc.

4. predict files from your terminal...<br>
    a. either individually without slurm:<br>
        `conda activate PpaPred` or `source activate PpaPred`<br>
        `python -u predict.py -p "pattern"`<br>
    b. individually with slurm:<br>
        `sbatch Predict.sh "pattern"`<br>
    c. or multiple with a slurm scripts (edit multiPredict.sh by providing name patterns of folders)<br>
    c. `sh multiPredict.sh`

#### If you want to use the provided jupyter notebooks for batch analysis:

1. Activate ipykernel for env (only once):<br>
    `conda activate PpaPred` or `source activate PpaPred`<br>
    Then `python -m ipykernel install --user --name PpaPred`

2. Edit config_batch.yml and provide patterns of files you want to analyse.

3. Connect to jupyter notebook and run BatchAnalysis_combined.ipynb step by step.<br>

### Happy Predicting :D

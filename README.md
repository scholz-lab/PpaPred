# PpaPred
### Prediction Pipeline for foraging behaviours of *Pristionchus pacificus* based on PharaGlow

#### Steps:
1. Download git repository:<br>
with ssh: `git clone git@github.com:scholz-lab/PpaPred.git`

2. Install environmnet:<br>
`conda env create -f requirements.txt -n PpaPred` or `conda env create -f environment.yml`

3. Edit config and config_batch files, to specifiy path to your PharaGlow files, etc.

4. predict files from your terminal...<br>
    a. either individually without slurm:<br>
        `conda activate PpaPred` or `source activate PpaPred`<br>
        `python -u predict.py -p "pattern"`<br>
    b. individually with slurm:<br>
        `sbatch Predict.sh "pattern"`<br>
    c. or multiple with slurm script (edit multiPredict.sh by providing name patterns of folders)<br>
    c. `sh multiPredict.sh`

#### If you want to use the provided jupyter notebook for batch analysis:

1. Activate ipykernel for env (only once):<br>
    `conda activate PpaPred` or `source activate PpaPred`<br>
    Then `python -m ipykernel install --user --name PpaPred`

2. Edit config_batch.yml and provide patterns of files you want to analyse.

3. Connect to jupyter notebook and run BatchAnalysis_combined.ipynb step by step.<br>

### Happy Predicting :D

## Folder Structure

This program uses patterns in the file and folder names to find files. Thus, the easiest way to handle predictions and pull data recorded at different days for comparison, is to have unique ids that can be used for identifcation: e.g. an id for the experiment (e.g. LB001), or specify condition and experiment id within the folder names (e.g. Exp01_WT, or LB001_WT). Alternatively, place files or folders in uniquely identifable folders, which can be specified as path in the config file.

## Update

If you want to pull changes from origin, you might encounter problems with BatchAnalysis_combined.ipynb. This is likely due to the output generated in this file. The easiest solution for this is to stash changes before pull.<br>
`git stash`<br>
`git pull`<br>
`git stash pop`<br>

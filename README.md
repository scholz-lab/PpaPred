# PpaPred
## Prediction Pipeline for foraging behaviours of *Pristionchus pacificus* based on PharaGlow

#### First steps:
1. Download git repository:<br>
`git clone git@github.com:scholz-lab/PpaPred.git`

2. Install environmnet:<br>
`conda env create -f requirements.txt -n PpaPred`

3. predict files from your terminal
`python -u predict.py -in "path/to/indir" -p "pattern" -o "path/to/outdir"`
If your lucky and you have HPC at your convenience, you can use the provided shell scripts.

If you want to use some of the provided jupyter notebooks
4. activate ipykernel for env:<br>
    `conda activate PpaPred` or `source activate PpaPred`<br>
    Then `python -m ipykernel install --user --name PpaPred`

4. open jupyter-notebook and predict :D<br>
Follow instructions in FeedingPrediction.ipynb. You will need to have results from PharaGlow.

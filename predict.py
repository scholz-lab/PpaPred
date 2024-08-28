# %% [markdown]
# # Prediction of Foraging States of *P. pacificus*
# 
# The single steps of this pipeline are the following:
# 1. additional feature calculation
# 2. pipeline loading
# 3. prediction
# 4. analysis and plotting

# %% IMPORTS

# lib imports
import os
import time
import tqdm
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import json
import joblib
import sklearn
from sklearn.impute import SimpleImputer
from scipy.stats.contingency import crosstab
import argparse

# custom imports
sys.path.append(os.path.expanduser('~'))
sys.path.append(os.getcwd())
from functions.load_model import load_tolist
import functions.process as proc
from functions.io import setup_logger, makedir
from functions.read_write import NpIntEncoder
from functions.FeatureEngine import CalculateFeatures 
from functions.plots_prediction import ethogram_plotter, CLtrajectory_plotter, transition_plotter
 
# %% SETTINGS
# set invalid (division by zero error) to ignore
np.seterr(invalid='ignore')
date = time.strftime("%Y%m%d")
datestr = time.strftime("%Y%m%d-%HH%MM")
newline = '\n'
sklearn.set_config(transform_output="pandas")

# %% INPUT
parser = argparse.ArgumentParser(description="Argument Parser", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-p", "--pattern", default=[''], nargs='+', type=str, help="input pattern (string): a pattern to find folder with data to predict.")
parser.add_argument("-id", "--jobid", default='xxxxxxxxxx', type=str, help="get jobid of slurm job")
args = vars(parser.parse_args())

inpath_pattern = args['pattern']
jobID = args['jobid']

# %% Configuration
config = yaml.safe_load(open("config.yml", "r"))

# pipeline
model_path = config['settings']['model']
version = os.path.basename(model_path).split("_")[1].split(".")[0]
ASpath = config['settings']['ASpath']
smooth = config['settings']['fbfill']

# recording
fps = config['settings']['fps']

# path
inpath = config['path']['PharaGlow data']
outpath = config['path']['predictions']
inpath_with_subfolders = config['path']['with subfolders']

# plots
summaries = config['analysis']['summaries']
ethograms = config['analysis']['ethograms']
transitions = config['analysis']['transitions']
trajectories = config['analysis']['trajectories']

# coloring and labels
cluster_color = config['cluster_color']
cluster_label = config['cluster_labels']
skip_already = config['settings']['skip_already']

feature_upper = {'velocity_dt60': 77} # 95 percentile TODO give possibility to either give 95 WT percentile or if not and to write on config

# %% I/O
# create list of inpath from folders in inpath that folder names contain inpath_pattern
if inpath_with_subfolders:
    new_inpath = [os.path.join(inpath, sub) for sub in os.listdir(inpath) if any(pat in sub for pat in inpath_pattern)]
    inpath = new_inpath
else:
    inpath = [inpath]

# make outpath
base_outpath = makedir(outpath)
outpath = []
for p in inpath:
    in_folder = os.path.basename(p)
    outpath.append(makedir(os.path.abspath(os.path.join(base_outpath,in_folder))))

# %% Logger Setup
logger_out = os.path.join(base_outpath,f"{jobID}_{datestr}_{inpath_pattern}_prediction.log")
logger = setup_logger('logger',filename=logger_out)

# first logger messages
logger.info(f"Foraging prediction of Pristionchus pacificus")
logger.info(f"Version of model == {version}, stored at {model_path}\n")
logger.info(f"Files to be predicted stored at:\n{newline.join(inpath)}")

# %% 1. Feature Engineering
# In the following section, additional features are calculated.
# The engineerd data files are saved under the specified outpath/subfolder.
# (with subfolder being the inpath folder name postfixed by _engine)
logger.info(f'\n')
logger.info(f'### Feature Engineering ###')

#TODO function that makes sure that fps is set to 30

FeatureEngine  = CalculateFeatures(inpath, 
                                    outpath, 
                                    logger, 
                                    return_XYCLine = True, 
                                    skip_engine = False, 
                                    skip_already=False, 
                                    out_fn_suffix='features',
                                    inpath_with_subfolders=inpath_with_subfolders)


XYs, CLines  = FeatureEngine.run()

all_engine = [os.path.join(root, name) for root, dirs, files in os.walk(base_outpath) for name in files if any(pat in os.path.basename(root) for pat in inpath_pattern) and name.endswith('features.json')]

# scaler to make mutants WT-like, to make comparable to WT phenotype
FeatureEngine.scale(wanted_quantile=feature_upper, is_quantile=.95, apply_to='outs')

# %% 2. Load Pipelines
model = joblib.load(open(model_path, 'rb'))
augsel = joblib.load(ASpath)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')

# %% 3. Prediction
logger.info(f'### Prediction ###')

prediction_done = []
notpredicted = []
# iterate over files on which feature engineering was performed
for fpath in tqdm.tqdm(all_engine):
    fn = os.path.basename(fpath)
    fn_dir = os.path.dirname(fpath)
    fn_out = os.path.basename(fpath).replace('features', 'prediction')

    # Check if necessary. renaming leads to invisibiltity of predicted files
    # TODO check if file was already predicted.
    # should work with sub folders and that files are renamed instead of new file created
    # if skip_already and fn_out in os.listdir(outpath):
    #    continue

    if not fn[0] == '.' and not fn in prediction_done and os.path.isfile(fpath):
        print(fn)
        d = load_tolist(fpath, droplabelcol=False)[0]
        
        # Data Augmentation
        X = augsel.transform(d) # augmentation pipeline
        X = imp.fit_transform(X) # simple imputer. imputes nan values

        # Ensure all features are in data
        # could be redundant with except statement
        if not X.shape[1] == model.n_features_in_:
            logger.info(f'WARNING: {fn} could not be predicted! Wrong number of features: got {X.shape[1]}, expected {model.n_features_in_}')
            notpredicted.append(fn)
            continue

        # predict
        try:
            pred = model.predict(X)
            proba = model.predict_proba(X)
        except Exception as error:
            logger.info(f'WARNING: {fn} could not be predicted!')
            logger.info(f'An exeption occured: {error}')
            notpredicted.append(fn)
            continue
        
        # reindex from fps=1 to original fps=30
        # minimla state duration is still 1 sec
        pred = pd.Series(pred, index=X.index, name='prediction').reindex(d.index, method='bfill', limit=29).fillna(-1) ### NEW
        proba = pd.DataFrame(proba, index=X.index, columns=[f'proba_{i}' for i in range(proba.shape[1])]).reindex(d.index, method='bfill', limit=29).fillna(0)
        
        # adjust prediction
        proba_max = np.amax(proba, axis=1) # probability should be same value over 30 frames / 1 sec
        proba_low50 = proba_max < .5 # find low probability predictions
        pred[proba_low50] = -1 # set predictions with low probability to -1 / None
        
        # output: append prediction and probaility to fpath
        p_out = pd.concat([d, pred, proba], axis=1) #d, 
        jsnL = json.loads(p_out.to_json(orient="split"))
        jsnF = json.dumps(jsnL, indent = 4)
        with open(fpath, "w") as outfile:
            outfile.write(jsnF)
        
        # rename fpath to clarify that it is predicted
        os.rename(fpath, os.path.join(fn_dir, fn_out))

logger.info(f'\n')
logger.info(f'Following files could not be predicted: {notpredicted}')

# %% 4. Analysis and Plotter
logger.info(f'\n')
logger.info(f'### Analysis ###')

all_predicted = [os.path.join(root, name) for root, dirs, files in os.walk(base_outpath) for name in files if any(pat in os.path.basename(root) for pat in inpath_pattern) and 'prediction.json' in name]

# iterate over predicted files
for fpath in tqdm.tqdm(all_predicted):
    fn = os.path.basename(fpath)
    fn_out = fn.replace('prediction.json','')

    fn_dir= os.path.dirname(fpath)
    out_analysis = makedir(os.path.join(fn_dir, 'analysis'))
    
    d = load_tolist(os.path.join(fpath), droplabelcol=False)[0]
    d['prediction'].to_csv(os.path.join(out_analysis, fn.replace('json','csv')), index=False) #TODO check if needed
    y = d['prediction'].values
    y = np.nan_to_num(y, -1)
    
    # get on and offsets from bouts
    onoff, dur, seque = proc.onoff_dict(y, labels =np.unique(y), return_duration=True, return_transitions=True)
    onoff = {int(k):v for k,v in onoff.items()}
    with open(os.path.join(out_analysis, fn_out+'_onoff.json'), "w") as onoff_out:  #TODO check if needed
        json.dump(onoff,onoff_out,cls=NpIntEncoder)

    if ethograms:            
        ethogram_plot = ethogram_plotter(d, y, onoff,  smooth, cluster_color, cluster_label, fn=fn)
        plt.savefig(os.path.join(out_analysis, fn_out+'_predictedbouts.pdf'))
        plt.close()

    if summaries:
        idx = pd.IndexSlice
        # create summaries for each state, each bout
        data_describe = d.groupby(y).describe().T.loc[idx[:, ['mean','std','count']], :].sort_index(level=0).T
        dur_describe = pd.DataFrame(dur, columns=['duration']).groupby(seque).describe().T.loc[idx[:, ['mean','std','count']], :].sort_index(level=0).T
        dur_describe['duration','relative'] = pd.DataFrame(dur, columns=['duration']).groupby(seque).apply(lambda cd: cd.sum()/len(d))
        
        # concat summaries
        summary = pd.concat([dur_describe, data_describe], axis=1)
        summary.index.name = 'cluster' # TODO check if easily changed to 'state'
        summary = summary.T.reset_index(drop=True).set_index(summary.T.index.map('_'.join)).T
        summary = summary.set_index(summary.index.astype(int))
        summary = summary.reindex([k for k in cluster_label if k != -1])
        summary.to_csv(os.path.join(out_analysis, fn_out+'summary.csv'))
    
    if transitions:
        # y downscaled to 1 fps, which is the minimum state duration
        y_1s = pd.DataFrame(y).rolling(30).apply(lambda s: s.mode()[0])[29::30].values.flatten()
        trans_col,fr_transition = crosstab(y_1s[1:], y_1s[:-1], levels=([k for k in cluster_label],[k for k in cluster_label]))
        transition_df = pd.DataFrame(fr_transition, columns = trans_col[1], index=trans_col[0])
        transition_df.to_csv(os.path.join(out_analysis, fn_out+'transitions.csv'))
        transition_norm = fr_transition/fr_transition.sum(axis=0)
        transition_norm_df = pd.DataFrame(transition_norm, columns = trans_col[1], index=trans_col[0])
        transition_norm_df.to_csv(os.path.join(out_analysis, fn_out+'transitions_norm.csv'))
    
        # TODO decide if leave or remove
        #transition_plot = transition_plotter(transition_all, cluster_color, node_alpha=summary['duration_relative'].fillna(0).tolist())
        #plt.savefig(os.path.join(out_analysis,fn_out+'clustertransitions.pdf'))
    

    if trajectories:
        XY = XYs[fn.replace('_prediction.json','.json_labeldata.csv')]
        CLine = CLines[fn.replace('_prediction.json','.json_labeldata.csv')]
        
        CLtrajectory_plot = CLtrajectory_plotter(CLine, XY, y, cluster_color, cluster_label, fn=fn, figsize=(10,10),)
        plt.savefig(os.path.join(out_analysis, fn_out+'CLtrajectory.pdf'))
        plt.close()

logger.info(f'\n')
logger.info(f'### End ###')
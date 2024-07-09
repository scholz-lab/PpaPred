# %% [markdown]
# # Prediction of Foraging States of *P. pacificus*
# 
# This notebook will guide you through the prediction pipeline for foraging behaviours in *Pristionchus pacificus*.<br>
# You will already need to have data that was extracted by PharaGlow.<br>
# 
# The single steps of this pipeline are the following:
# 1. additional feature calculation
# 2. model and augmentation loading
# 3. data augmentation as defined by AugmentSelect file
# 4. prediction
# 5. visualisation

# %%
import os
import time
import tqdm
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyArrowPatch
import logging
import yaml
import json
import joblib
from sklearn.impute import SimpleImputer
from scipy.stats.contingency import crosstab
import networkx as nx
from matplotlib.lines import Line2D
import umap
import itertools
from sklearn.preprocessing import power_transform

#home = os.path.expanduser("~")
sys.path.append(os.getcwd())
from functions.load_model import load_tolist
import functions.visualise as vis
import functions.process as proc
from functions.io import setup_logger, makedir
from functions import FeatureEngine
sys.path.append(os.path.expanduser('~'))
from PpaPy.processing.preprocess import addhistory, select_features
from functions.modelfunctions import add_power_transform, select_features, addhistory
import argparse

import pickle
from sklearn import set_config
 
from numba import jit
# set invalid (division by zero error) to ignore
np.seterr(invalid='ignore')


class NpIntEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

class NanConverter(json.JSONEncoder):
    def nan2None(self, obj):
        if isinstance(obj, dict):
            return {k:self.nan2None(v) for k,v in obj.items()}
        elif isinstance(obj, list):
            return [self.nan2None(v) for v in obj]
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        return obj
    def encode(self, obj, *args, **kwargs):
        return super().encode(self.nan2None(obj), *args, **kwargs)
    
# %% [markdown]
# Please provide where your files are stored and where you would like your data to be saved in the following section.

parser = argparse.ArgumentParser(description="Argument Parser",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-in", "--inpath", default="", type=str, help="provide the inpath to the dataset")
parser.add_argument("-p", "--pattern", default=[''], nargs='+', type=str, help="Build model from pieces: scaler: MinMaxScaler, to_dask: numpy array to dask, RF: RandomForestClassifier, PCA: pca, fs_prefit: feature selection based on SelectKBest")
parser.add_argument("-o", "--outpath", default="", type=str, help="provide the outpath where data should be safed")
parser.add_argument("-id", "--jobid", default='xxxxxxxxxx', type=str, help="get jobid of slurm job")


args = vars(parser.parse_args())
# %%
inpath = args['inpath']
inpath_with_subfolders = True
inpath_pattern = args['pattern']
args_out = args['outpath']
jobID = args['jobid']

base_outpath = makedir(args_out)

# %%
date = time.strftime("%Y%m%d")
datestr = time.strftime("%Y%m%d-%HH%MM")
home = os.path.expanduser("~")

if inpath_with_subfolders:
    new_inpath = [os.path.join(inpath, sub) for sub in os.listdir(inpath) if any(pat in sub for pat in inpath_pattern)]
    inpath = new_inpath
else:
    inpath = [inpath]

outpath = []
for p in inpath:
    in_folder = os.path.basename(p)
    outpath.append(makedir(os.path.abspath(f"{base_outpath}/{in_folder}")))


# %%
# In the following section, standard model parameters are set. Change those only if necessary.
# changes to config file are preferrerable
config = yaml.safe_load(open("config.yml", "r"))

cluster_color = config['cluster_color']
cluster_group = config['cluster_group_man']
cluster_label = config['cluster_names']
clu_group_label = {_:f'{_}, {__}' for _, __ in tuple(zip([c for c in cluster_label.values()],[g for g in cluster_group.values()]))}
skip_already = config['settings']['skip_already']
overwrite = True

model_path = config['settings']['model']
version = os.path.basename(model_path).split("_")[1].split(".")[0]
ASpath = config['settings']['ASpath']
smooth = config['settings']['fbfill']
fps = config['settings']['fps']

# lists to store already processed files in
prediction_done = []

# logger file created
logger_out = os.path.join(base_outpath,f"{jobID}_{datestr}_{inpath_pattern}_prediction.log")
logger = setup_logger('logger',filename=logger_out)
logger.info(f"Foraging prediction of Pristionchus pacificus")
logger.info(f"Version of model == {version}, stored at {model_path}\n")
log_inpath = '\n'.join(inpath)
logger.info(f"Files to be predicted stored at:\n{log_inpath}")

XYs, CLines  = FeatureEngine.run(inpath, outpath, logger, return_XYCLine =True, skip_engine = True, skip_already=False, out_fn_suffix='features') # skip_engine skip_already

# %%
def transition_plotter(transition_toother, cluster_color, transition_self=None, figsize=(8,6), mut_scale=40, node_size=4000, 
                    other_connectionstyle = "arc3,rad=.15", self_connectionstyle="arc3,rad=0.5", node_alpha = 1, exclude_label = [], clu_group_label=None):
    if transition_self is None:
        #print(transition_self)
        transition_self = transition_toother.copy().diagonal()
        np.fill_diagonal(transition_toother, 0)


        
    A = np.nan_to_num(np.around(transition_toother.T,3))
    G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    
    weights = nx.get_edge_attributes(G,'weight').values()
    arr_out = [e[0] for e in G.edges(data=True)]
                        
    color_map = [cluster_color[k] for k in cluster_color if k not in exclude_label]
    edge_color = [cluster_color[c-1] for c in arr_out]
    #edge_alpha = [node_alpha[c] for c in arr_out]
                        
    fig, ax = plt.subplots(1, figsize=figsize)
    fig_w = fig.get_size_inches()[0]
    arrowsize = [w*mut_scale for w in weights]
                        
    if clu_group_label is None:
        labels = dict(zip(range(len(G)),range(len(G))))
    else:
        labels = dict(zip(range(len(G)),  [clu_group_label[k] for k in clu_group_label if k not in exclude_label]))
        
    label_collection = nx.draw_networkx_labels(G, pos=nx.circular_layout(G), ax=ax, labels=labels)
    
    node_collection = nx.draw_networkx_nodes(G, pos=nx.circular_layout(G), ax=ax, node_color = color_map, node_size= node_size, margins=0.1,
                                             alpha= node_alpha,
                                             edgecolors=color_map)
    edge_collection = nx.draw_networkx_edges(G, pos=nx.circular_layout(G), ax=ax, 
                                             arrowsize =arrowsize, connectionstyle=other_connectionstyle, arrowstyle="simple",
                                             label=list(weights), node_size=node_size, edge_color=edge_color)
    
    ### to self
    edgelist = [i for i in G.nodes() if transition_self[i] > 0 and transition_self[i] != np.nan]
    selfweights = {i:transition_self[i] for i in edgelist}
    G.add_edges_from([(i,i) for i in edgelist])
    
    for i in edgelist:
        cor = np.round(nx.circular_layout(G)[i],2)
        rad = np.arctan2(*cor)-np.arctan2(0,0)
        rad_s, rad_t = rad-.15, rad+.15
        vl = np.linalg.norm(cor)+.2
        xy_t = [vl*np.sin(rad_s),vl*np.cos(rad_s)]
        xy_s = [vl*np.sin(rad_t),vl*np.cos(rad_t)]
        (A, _, C, D) =  vis.SemiCirc_coordinates(xy_s, xy_t, r=0.2)
        arrow0 = FancyArrowPatch(posA=A, posB=D, connectionstyle=self_connectionstyle, arrowstyle="simple", mutation_scale= selfweights[i]*mut_scale, color=color_map[i])
        arrow1 = FancyArrowPatch(posA=D, posB=C, connectionstyle=self_connectionstyle, arrowstyle="simple", mutation_scale= selfweights[i]*mut_scale, color=color_map[i])
        ax.add_artist(arrow0)
        ax.add_artist(arrow1)
    
        
    for arr_s in np.linspace(0.2,1,5):
        arrow = FancyArrowPatch((1.6, arr_s), (1.9, arr_s), mutation_scale=arr_s*mut_scale, label = arr_s, color='k', alpha=0.5)
        ax.text(1.95, arr_s-0.03, f"{int(arr_s*100)}%")
        ax.add_patch(arrow)
    ax.set_xlim(-2,2)
    ax.set_ylim(-1.5,1.5)
    ax.axis('off')
    return fig

def ethogram_plotter(d, y, onoff,  smooth, cluster_color, figsize=(20,5), fps=30,xtick_spread=30, d_toplot=['velocity', 'rate'], d_bar_alpha =0.3):
    timeinsec = np.array(range(len(d)))/fps
    
    fig, axs = plt.subplots(3,1, figsize=figsize,constrained_layout=True)
    
    for c in np.unique(y).astype(int):
        axs[0].broken_barh(onoff[c],(0,1),facecolors = cluster_color[c])
    axs[0].set_xticks(range(len(timeinsec))[::xtick_spread*fps])
    axs[0].set_xticklabels(timeinsec[::xtick_spread*fps].astype(int))
    axs[0].set_title(f'Cluster preditcion (smoothed {smooth/fps} sec).')
    axs[0].xaxis.set_minor_locator(plt.MultipleLocator(5*fps))
    for i,c in enumerate(d_toplot):
        for c_ in np.unique(y).astype(int):
            #axs[i+1].broken_barh(onoff[c_],(min(d[c]),max(d[c])),facecolors = cluster_color[c_], alpha=0.6, zorder=0)
            axs[i+1].broken_barh(onoff[c_],(np.nanmin(d[c]),np.nanmax(d[c])-np.nanmin(d[c])),facecolors = cluster_color[c_], alpha=d_bar_alpha, zorder=0)
        axs[i+1].plot(d[c].rolling(30, min_periods=0).mean(),c='k')
        axs[i+1].set_xticks(range(len(timeinsec))[::xtick_spread*fps])
        axs[i+1].set_xticklabels(timeinsec[::xtick_spread*fps].astype(int))
        axs[i+1].set_title(f"{c} (smoothed, 1 sec)")
        axs[i+1].xaxis.set_minor_locator(plt.MultipleLocator(5*fps))
    axs[2].set_xlabel('sec')
    
    plt.legend(handles=[Patch(facecolor=cluster_color[i]) for i in np.unique(y).astype(int)],
          labels=[clu_group_label[k] for k in cluster_label if k in np.unique(y)],
          ncol=3, loc='upper left',
          bbox_to_anchor=(0, -0.5))
    fig.suptitle(f'Ethogram of {fn}',fontsize=16)
    return fig
    
def CLtrajectory_plotter(CLine, XY, y, cluster_color, cluster_label, figsize=(10,10)):
    fig, ax = plt.subplots(figsize=(10,10))
    legend_elements = [Line2D([0], [0],color=cluster_color[i], label=cluster_label [i]) for i in cluster_label]
    adjustCL = (CLine-np.nanmean(CLine))+np.repeat(XY.reshape(XY.shape[0],1,XY.shape[1]), CLine.shape[1], axis=1)-np.nanmean(XY, axis=0)# fits better than subtracting 50
    adjustXY = XY-np.nanmean(XY, axis=0)
    for l in np.unique(y).astype(int):
    #for l in [2,3,5,8]:#[1,2,6,7]#[2,3,5,8]
        #if l != 6:
        il = np.where(y == l)[0]
        ax.plot(*adjustCL[il].T, c=cluster_color[l], alpha = 0.1)#cluster_color[l]
            #plt.scatter(XY[:,0][il],XY[:,1][il], marker=".", lw=2, c=bar_c[l], alpha=0.1)
    ax.set_title(fn)
    ax.axis('equal')
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1,1))
    return fig

# %%
# %%
ethograms = True
summaries = True
transitions = True
trajectories = True

all_predicted = [os.path.join(root, name) for root, dirs, files in os.walk(base_outpath) for name in files if any(pat in os.path.basename(root) for pat in inpath_pattern) and 'prediction.json' in name]

# %%
logger.info(f'\n')
logger.info(f'### Analysis ###')
for fpath in tqdm.tqdm(all_predicted):
    fn = os.path.basename(fpath)
    fn_out = fn.replace('prediction.json','')

    dir_engine = os.path.dirname(fpath)
    out_analysis = makedir(os.path.join(dir_engine, 'analysis'))
    
    d = load_tolist(os.path.join(fpath), droplabelcol=False)[0]
    d['prediction'].to_csv(os.path.join(out_analysis, fn.replace('json','csv')), index=False)
    y_ps = d['prediction'].values
    y_ps = np.nan_to_num(y_ps, -1)
        
    if ethograms:            
        onoff = proc.onoff_dict(y_ps, labels = np.unique(y_ps))
        onoff = {int(k):v for k,v in onoff.items()}
        with open(os.path.join(out_analysis, fn_out+'_onoff.json'), "w") as onoff_out: 
            json.dump(onoff,onoff_out,cls=NpIntEncoder)
        ethogram_plot = ethogram_plotter(d, y_ps, onoff,  smooth, cluster_color)
        #plt.savefig('clusterbouts.pdf')
        plt.savefig(os.path.join(out_analysis, fn_out+'_predictedbouts.pdf'))
        plt.close()

    if summaries:
        idx = pd.IndexSlice
        onoff, dur, transi = proc.onoff_dict(y_ps, labels =np.unique(y_ps), return_duration=True, return_transitions=True)
        data_describe = d.groupby(y_ps).describe().T.loc[idx[:, ['mean','std','count']], :].sort_index(level=0).T
        dur_describe = pd.DataFrame(dur, columns=['duration']).groupby(transi).describe().T.loc[idx[:, ['mean','std','count']], :].sort_index(level=0).T
        dur_describe['duration','relative'] = pd.DataFrame(dur, columns=['duration']).groupby(transi).apply(lambda cd: cd.sum()/len(d))
        summary = pd.concat([dur_describe, data_describe], axis=1)
        summary.index.name = 'cluster'
        summary = summary.T.reset_index(drop=True).set_index(summary.T.index.map('_'.join)).T
        summary = summary.set_index(summary.index.astype(int))
        summary = summary.reindex([k for k in cluster_label if k != -1])
        summary.to_csv(os.path.join(out_analysis, fn_out+'summary.csv'))
    
    if transitions:
        y_ps_transition = pd.DataFrame(y_ps).rolling(30).apply(lambda s: s.mode()[0])[29::30].values.flatten()
        
        trans_col,fr_transition = crosstab(y_ps_transition[1:],y_ps_transition[:-1],
                                           levels=([k for k in cluster_label],[k for k in cluster_label])
                                          )
        #othersum_axis0 = fr_transition.sum(axis=0)-fr_transition.diagonal()
        transition_all = fr_transition/fr_transition.sum(axis=0)
        #transition_toother = fr_transition/othersum_axis0
        #transition_self = fr_transition.diagonal()/(fr_transition.sum(axis=0))
        #np.fill_diagonal(transition_toother, 0)
        
        #transition_merged = transition_toother.copy()
        #diag_idx = np.diag_indices(len(transition_merged))
        #transition_merged[diag_idx] = transition_self
        transition_merged = pd.DataFrame(transition_all, columns = trans_col[1], index=trans_col[0])#.fillna(0) #should not fill nan with 0!
        transition_merged.to_csv(os.path.join(out_analysis, fn_out+'transitions.csv'))
        
        #### TRANSITION PLOT
    
        #transition_plot = transition_plotter(transition_all, cluster_color, node_alpha=summary['duration_relative'].fillna(0).tolist())
        #plt.savefig(os.path.join(out_analysis,fn_out+'clustertransitions.pdf'))
    

    if trajectories:
        XY = XYs[fn.replace('_prediction.json','.json_labeldata.csv')]
        CLine = CLines[fn.replace('_prediction.json','.json_labeldata.csv')]


        
        CLtrajectory_plot = CLtrajectory_plotter(CLine, XY, y_ps, cluster_color, cluster_label, figsize=(10,10),)
        plt.savefig(os.path.join(out_analysis, fn_out+'CLtrajectory.pdf'))
        plt.close()

logger.info(f'\n')
logger.info(f'### End ###')
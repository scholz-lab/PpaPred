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
 
from numba import jit
# set invalid (division by zero error) to ignore
np.seterr(invalid='ignore')

# %% [markdown]
# Please provide where your files are stored and where you would like your data to be saved in the following section.

parser = argparse.ArgumentParser(description="Argument Parser",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-in", "--inpath", default="", type=str, help="provide the inpath to the dataset")
parser.add_argument("-p", "--pattern", default=[''], nargs='+', type=str, help="Build model from pieces: scaler: MinMaxScaler, to_dask: numpy array to dask, RF: RandomForestClassifier, PCA: pca, fs_prefit: feature selection based on SelectKBest")
parser.add_argument("-o", "--outpath", default="", type=str, help="provide the outpath where data should be safed")


args = vars(parser.parse_args())
# %%
def add_power_transform(df, cols):
    arr_t = power_transform(df.loc[:,cols])
    return pd.concat([df, pd.DataFrame(arr_t, columns = [c+'_tr' for c in cols])], axis = 1)

def select_features(df, names):

    if isinstance(names, str):
        if names == 'all':
            names = df.columns
    return df.loc[:, names]

def addhistory(df, dt_shifted, name_filter=None):
    """
    Concats history along axis 1 of df.
    Args: 
        df (DataFrame): dataframe across which to append history
        dt_shifted (list): list of absolute frame values to go back and forward in time, for each frame
        name_filter (list): column names that should not be included in addhistory, default None
    Returns:
        multshift (dataframe): dataframe with history, history columns have suffic _pos or _neg
    """
    if name_filter is not None:
        name_filter_all = list(df.filter(regex="|".join(name_filter)).columns)
        name_keep = df.columns.drop(name_filter_all)
        df_ = df.loc[:, name_filter_all]
        df = df.loc[:, name_keep]
    multshift = df.copy()
    for i in dt_shifted:
        p_shift = df.shift(i)
        n_shift = df.shift(-i)
        multshift = pd.concat([multshift, p_shift.add_suffix(f"_pos{i}"), n_shift.add_suffix(f"_neg{i}")], axis=1)
    if name_filter is not None:
        multshift = pd.concat([df_, multshift], axis=1)
    
    return multshift 
# %%
#args = {'inpath': "/gpfs/soma_fs/scratch/src/boeger/data_gueniz/",
#        'pattern': ["Exp1_WT_larvae"],
#        'outpath': "/gpfs/soma_fs/scratch/src/boeger/PpaPred_eren_2403"}
inpath = args['inpath']
inpath_with_subfolders = True
inpath_pattern = args['pattern']
args_out = args['outpath']


base_outpath = makedir(args_out) #makedir('/gpfs/soma_fs/scratch/src/boeger/PpaPred_eren')
#base_outpath = makedir('/gpfs/soma_fs/scratch/src/boeger/data_roca')

# %%
date = time.strftime("%Y%m%d")
datestr = time.strftime("%Y%m%d-%HH%MM")
home = os.path.expanduser("~")

if inpath_with_subfolders:
    new_inpath = [os.path.join(inpath, sub) for sub in os.listdir(inpath) if any(pat in sub for pat in inpath_pattern)]
    inpath = new_inpath
else:
    inpath = [inpath]

outpath, out_engine, out_predicted = [],[],[]
for p in inpath:
    in_folder = os.path.basename(p)
    outpath.append(makedir(os.path.abspath(f"{base_outpath}/{in_folder}"))) # you can also use datestr to specify the outpath folder, like this makedir(os.path.abspath(f"{datestr}_PpaPrediction"))
    out_engine.append(os.path.join(outpath[-1], in_folder+'_engine'))
    #out_predicted.append(os.path.join(outpath[-1], in_folder+'_predicted'))

# %%
os.path.commonpath(inpath)

# %% [markdown]
# In the following section, standard model parameters are set. Change those only if necessary.

# %%
config = yaml.safe_load(open("config.yml", "r"))

# %%
cluster_color = config['cluster_color']
cluster_group = config['cluster_group_man']
cluster_label = config['cluster_names']
clu_group_label = {_:f'{_}, {__}' for _, __ in tuple(zip([c for c in cluster_label.values()],[g for g in cluster_group.values()]))}
skip_already = config['settings']['skip_already']

# %%
model_path = config['settings']['model']
version = os.path.basename(model_path).split("_")[1].split(".")[0]
ASpath = config['settings']['ASpath']
smooth = config['settings']['fbfill']
fps = config['settings']['fps']
engine_done = []
prediction_done = []

logger_out = os.path.join(base_outpath,f"{datestr}_PpaForagingPrediction.log")
logger = setup_logger('logger',filename=logger_out)
logger.info(f"Foraging prediction of Pristionchus pacificus")
logger.info(f"Version of model == {version}, stored at {model_path}\n")
log_inpath = '\n'.join(inpath)
logger.info(f"Files to be predicted stored at:\n{log_inpath}")

# %% [markdown]
# ## 1. Feature Engineering
# In the following section, additional features are calculated.<br>
# The engineerd data files are saved under the specified outpath/subfolder.<br>
# (with subfolder being the inpath folder name postfixed by _engine)

# %%
XYs, CLines  = FeatureEngine.run(inpath, out_engine, logger, return_XYCLine =True, skip_engine = False, skip_already=False)

# %%
import pickle
model = joblib.load(open(model_path, 'rb'))
augsel = joblib.load(ASpath)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')

# %%
all_engine = [os.path.join(root, name) for root, dirs, files in os.walk(base_outpath) for name in files if 'engine' in os.path.basename(root) and any(pat in os.path.basename(root) for pat in inpath_pattern)]
all_engine

# %% [markdown]
# ## 3. Prediction

# %%
skip_already = False
for fpath in tqdm.tqdm(all_engine):
    fn = os.path.basename(fpath)
    dir_engine = os.path.dirname(fpath)
    out_predicted = makedir(dir_engine[:-len('engine')]+'predicted')
    out_fn = fn.replace('features', 'predicted')
    if skip_already and out_fn in os.listdir(out_predicted):
        continue
    if not fn[0] == '.' and not out_fn in prediction_done and os.path.isfile(fpath):
        d = load_tolist(fpath, droplabelcol=False)[0]
        
        X = augsel.fit_transform(d)
        col = X.columns
        X = pd.DataFrame(imp.fit_transform(X), columns = col)
        
        pred = model.predict(X)
        proba = model.predict_proba(X)
        pred_smooth = proc.ffill_bfill(pred, smooth)
        pred_smooth = np.nan_to_num(pred_smooth,-1)
        proba_max = np.amax(proba, axis=1) ### New
        proba_max_mean = pd.DataFrame(proba_max).rolling(30, min_periods=1).mean().values ### New
        proba_low = np.all(proba_max_mean < .5, axis=1) ### New
        pred_smooth[proba_low] = -1 ### NEW

        #fn = os.path.basename(fn)
        #out_fn = '_'.join(fn.split('_')[:4]+['predicted.json'])
        p_out = pd.concat([d, pd.DataFrame(pred_smooth, columns=['prediction']), pd.DataFrame(model.predict_proba(X), columns=[f'proba_{i}' for i in range(proba.shape[1])])], axis=1)

        jsnL = json.loads(p_out.to_json(orient="split"))
        jsnF = json.dumps(jsnL, indent = 4)
        outpath_p = os.path.join(out_predicted,out_fn)
        with open(outpath_p, "w") as outfile:
            outfile.write(jsnF)
        
# %% [markdown]
# ## 4. Prediction
# The augmented + predicted data files are saved under the specified outpath/subfolder.<br>
# (with subfolder being the inpath folder name postfixed by _predicted)<br>
# 
# In the _predicted, plots of the bouts predicted over time along with the velocity and pumping rate are saved as pdf files.

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
class NpIntEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

# %%
ethograms = True
summaries = True
transitions = True
trajectories = True

all_predicted = [os.path.join(root, name) for root, dirs, files in os.walk(base_outpath) for name in files if 'predicted' in os.path.basename(root) and any(pat in os.path.basename(root) for pat in inpath_pattern) and 'predicted.json' in name]
#for fn in tqdm.tqdm(os.listdir(out_predicted)):
#    if fn[-13:] == 'predicted.csv' or fn[-14:] == 'predicted.json':

# %%
len(all_predicted)

# %%
for fpath in tqdm.tqdm(all_predicted):
    
    fn = os.path.basename(fpath)
    fn_out = fn.replace('predicted.json','')
    out_predicted = os.path.dirname(fpath)
    
    d = load_tolist(os.path.join(out_predicted,fn), droplabelcol=False)[0]
    y_ps = d['prediction'].values
    d['prediction'].to_csv(os.path.join(out_predicted, fn_out+'prediction.csv'), index=False)
        
    if ethograms:            
        onoff = proc.onoff_dict(y_ps, labels =np.unique(y_ps))
        onoff = {int(k):v for k,v in onoff.items()}
        with open(os.path.join(out_predicted, fn_out+'_onoff.json'), "w") as onoff_out: 
            json.dump(onoff,onoff_out,cls=NpIntEncoder)
        ethogram_plot = ethogram_plotter(d, y_ps, onoff,  smooth, cluster_color)
        #plt.savefig('clusterbouts.pdf')
        plt.savefig(os.path.join(out_predicted, fn_out+'_predictedbouts.pdf'))
        plt.show()

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
        summary.to_csv(os.path.join(out_predicted, fn_out+'summary.csv'))
    
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
        transition_merged.to_csv(os.path.join(out_predicted, fn_out+'transitions.csv'))
        
        #### TRANSITION PLOT
    
        #transition_plot = transition_plotter(transition_all, cluster_color, node_alpha=summary['duration_relative'].fillna(0).tolist())
        #plt.savefig(os.path.join(out_predicted,fn_out+'clustertransitions.pdf'))
    

    if trajectories:
        XY = XYs[fn.replace('_predicted.json','.json_labeldata.csv')]
        CLine = CLines[fn.replace('_predicted.json','.json_labeldata.csv')]


        
        CLtrajectory_plot = CLtrajectory_plotter(CLine, XY, y_ps, cluster_color, cluster_label, figsize=(10,10),)
        plt.savefig(os.path.join(out_predicted, fn_out+'CLtrajectory.pdf'))


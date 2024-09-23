import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.lines import Line2D

from scipy.cluster.hierarchy import dendrogram
import networkx as nx

from . import process as proc



def SemiCirc_coordinates(start, end, r=None):
    start = np.array(start)
    end = np.array(end)
    half_StartEnd = (np.linalg.norm(start-end)/2) #calculate half length of start to end
    if r==None or r < half_StartEnd: # if the radius desired is smaller than the half_StartEnd, reset radius to half_StartEnd
        r=half_StartEnd
    a = np.sqrt(abs(half_StartEnd**2 - r**2)) # calculate the length of the midpoint to point halfway on circle
    alpha = np.arctan(a/half_StartEnd)
    end_2start = np.arctan2(*end-start)
    B_2origin = alpha + end_2start
    B = r*np.sin(B_2origin)+start[0],r*np.cos(B_2origin)+start[1]
    
    arc_of_arrow = np.pi-(np.pi-alpha*2)/2 # radians
    startD = (r * np.sin(arc_of_arrow/2))*2 #length
    rad_BstartD = (np.pi-arc_of_arrow)/2#
    radstartB_toX = np.arctan2(*B-start)
    rad_startD = rad_BstartD+radstartB_toX
    D = startD*np.sin(rad_startD)+start[0],startD*np.cos(rad_startD)+start[1] # coordinate 2

    return (start, B, end, D)

def transition_plotter(transition_toother, cluster_color, transition_self=None, figsize=(8,6), mut_scale=40, node_size=4000, edge_alpha=1,
                    other_connectionstyle = "arc3,rad=.15", self_connectionstyle="arc3,rad=0.5", node_alpha = 1, exclude_label = [-1], clu_group_label=None):
    """
    Returns figure for transitions visualised as in network x plot, behaviors visualised as circles, transitions as arrows.
    Args:
        transition_toother (ndarray): of size (m,m.T)
        cluster_color (dict): colormap as dictionary, keys responding to y labels
        transition_self (ndarray or None): if not None, array containing self transitions
        mut_scale (int): mutation scale
        node_size (int): size of nodes
        other_connectionstyle (string): connectionstyle to other nodes, string as defined by matplotlib
        self_connectionstyle (string): connectionstyle to self, string as defined by matplotlib
        node_alpha (float, list): alpha values for nodes
        edge_alpha (float, ndarray): alpha values for nodes
        exclude_label (list): labels in transition_toother and transition_self to exclude from plotting
        clu_group_label (dict): labels for clusters
    Returns:
        fig (matplotlib.figure): transition figure
    """
    
    # if not provided, get self transitions from transition_toother 
    if transition_self is None:
        print(transition_self)
        transition_self = transition_toother.copy().diagonal()
        np.fill_diagonal(transition_toother, 0)
        
    A = np.nan_to_num(np.around(transition_toother.T,3))
    # to do what is this doing
    if tuple(map(int, (nx.__version__.split(".")))) < tuple(map(int, ('3'.split(".")))):
        G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    else:
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    
    # get weights from networkx object
    weights = nx.get_edge_attributes(G,'weight').values()
    # get arrows out
    arr_out = [e[0] for e in G.edges(data=True)]

    # create lists of colors
    color_map = [cluster_color[k] for k in cluster_color if k not in exclude_label]
    edge_color = [cluster_color[c] for c in arr_out]
    #edge_alpha = [node_alpha[c] for c in arr_out]

    # create figure               
    fig, ax = plt.subplots(1, figsize=figsize)
    # calculate arrow sizes by multiplication with mut_scale
    arrowsize = [w*mut_scale for w in weights]

    # if not provided, infer labels from data                
    if clu_group_label is None:
        labels = dict(zip(range(len(G)),range(len(G))))
    else:
        labels = dict(zip(range(len(G)),  [clu_group_label[k] for k in clu_group_label if k != -1]))
    
    # draw labels
    label_collection = nx.draw_networkx_labels(G, pos=nx.circular_layout(G), ax=ax, labels=labels)
    
    # draw nodes
    node_collection = nx.draw_networkx_nodes(G, pos=nx.circular_layout(G), ax=ax, node_color=color_map, node_size= node_size, margins=0.1,
                                             alpha=node_alpha, edgecolors=color_map)
    
    # draw edges
    edge_collection = nx.draw_networkx_edges(G, pos=nx.circular_layout(G), ax=ax, 
                                             arrowsize=arrowsize, connectionstyle=other_connectionstyle, arrowstyle="simple",
                                             label=list(weights), node_size=node_size, edge_color=edge_color, alpha=edge_alpha)
    
    ### add edges to self
    self_edges = [i for i in G.nodes() if transition_self[i] > 0 and transition_self[i] != np.nan]
    self_weights = {i:transition_self[i] for i in self_edges}
    G.add_edges_from([(i,i) for i in self_edges])
    # custom design for self loops
    for i in self_edges:
        cor = np.round(nx.circular_layout(G)[i],2)
        rad = np.arctan2(*cor)-np.arctan2(0,0)
        rad_s, rad_t = rad-.15, rad+.15
        vl = np.linalg.norm(cor)+.2
        xy_t = [vl*np.sin(rad_s),vl*np.cos(rad_s)]
        xy_s = [vl*np.sin(rad_t),vl*np.cos(rad_t)]
        (A, _, C, D) =  SemiCirc_coordinates(xy_s, xy_t, r=0.2)
        arrow0 = FancyArrowPatch(posA=A, posB=D, connectionstyle=self_connectionstyle, arrowstyle="simple", mutation_scale= self_weights[i]*mut_scale, color=color_map[i])
        arrow1 = FancyArrowPatch(posA=D, posB=C, connectionstyle=self_connectionstyle, arrowstyle="simple", mutation_scale= self_weights[i]*mut_scale, color=color_map[i])
        ax.add_artist(arrow0)
        ax.add_artist(arrow1)
    
    # put legend for arrow size  
    for arr_s in np.linspace(0.2,1,5):
        arrow = FancyArrowPatch((1.6, arr_s), (1.9, arr_s), mutation_scale=arr_s*mut_scale, label = arr_s, color='k', alpha=0.5)
        ax.text(1.95, arr_s-0.03, f"{int(arr_s*100)}%")
        ax.add_patch(arrow)

    # some settings
    ax.set_xlim(-2,2)
    ax.set_ylim(-1.5,1.5)
    ax.axis('off')
    return fig

def transitions_plotter_percentage(transition_matrix, node_probability, cluster_color, label, edge_prob_thresh=0, maxout_nodeprob=True):
    trans = transition_matrix.copy()

    edge_alpha = trans.T.values.flatten()
    edge_alpha = edge_alpha[edge_alpha>0]
    edge_alpha[edge_alpha >= edge_prob_thresh] = 1
    edge_alpha[edge_alpha < edge_prob_thresh] = edge_prob_thresh

    node_label = (((node_probability*100).round().astype(int)).astype(str)+ '%').to_dict()

    if maxout_nodeprob:
        if isinstance(maxout_nodeprob, float):
            node_probability = node_probability/maxout_nodeprob
        else:
            node_probability = node_probability/np.max(node_probability.values)

    transition_plot = transition_plotter(trans.values.copy(), cluster_color,
                                         node_alpha=node_probability, edge_alpha=edge_alpha.T, clu_group_label=node_label)
    plt.title(f"transitions of {label}")
    return transition_plot

def ethogram_plotter(d, y, onoff,  smooth, cluster_color, cluster_label, fn='', figsize=(20,5), fps=30,xtick_spread=30, d_toplot=['velocity', 'rate'], d_bar_alpha =0.3):
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
          labels=[cluster_label[k] for k in np.unique(y)],
          ncol=3, loc='upper left',
          bbox_to_anchor=(0, -0.5))
    fig.suptitle(f'Ethogram of {fn}',fontsize=16)
    return fig
    
def CLtrajectory_plotter(CLine, XY, y, cluster_color, cluster_label, fn='', figsize=(10,10)):
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

def sig_stars(p):
    """
    Returns number of significant asterices from p value.
    Args:
        p (float): p value
    Returns:
        number of asterices (int)
    """
    if p <= .01:
        return min(int(-np.ceil(np.log10(abs(p)))),4)
    elif p <= .05:
        return 1
    else:
        return 0

def find_common(*lists):
    common = [value for value in lists[0] if all(value in lst for lst in lists)]
    return common

def find_uncommon(*lists):
    all_items = np.array(lists)
    df_items = pd.DataFrame(lists)
    common = find_common(*all_items)
    common_bool = np.isin(df_items,common)
    uncommon = []
    for i,col in enumerate(common_bool):
        col_uncommon = df_items.loc[i,~col]
        uncommon.append('_'.join(col_uncommon.dropna()))
    return uncommon

class StateConditionBoxplot():
    def __init__(self, multi_df, color_dict,  stats_df=None, y_label='', bonferroni=False, multi_level = 0, plot_percentile=(0,100), 
                 showfliers=True, showlegend=False, adaptive_figsize=True, cluster_label=None, figsize=(6,4), y_order={0:2,1:1,2:0,3:3,4:4,5:5}):
        self.multi_df = multi_df
        self.stats_df = stats_df
        self.plot_percentile = plot_percentile
        self.nan_percentiles()
        self.multi_level = multi_level
        self.bonferroni = bonferroni
        self.p_to_use = 'p' if not self.bonferroni else 'bonferroni p'
        self.y_label = y_label
        self.y_order = y_order
        
        self.showfliers = showfliers
        self.showlegend = showlegend
        self.adaptive_figsize = adaptive_figsize #True
        self.figsize = figsize #(6,4)
        self.color_dict = color_dict
        self.cluster_label = cluster_label if cluster_label is not None else {c: c for c in color_dict.keys()}

        self.conditions = self.multi_df.columns.get_level_values(self.multi_level).unique()
        

    def nan_percentiles(self):
        self.percentile = (np.round(np.nanpercentile(self.multi_df,self.plot_percentile[0])).astype(int),np.round(np.nanpercentile(self.multi_df,self.plot_percentile[1])).astype(int))

    def plot(self):
        if self.adaptive_figsize:
            self.figsize = (self.figsize[0]*len(self.conditions), self.figsize[1])
        self.index_plot = []              
        fig, ax = plt.subplots(figsize=self.figsize)
            
        for i, cond in enumerate(self.conditions):
            for j,c in enumerate(self.multi_df.index):
                if isinstance(c, str):
                    c_ = eval(c)
                else:
                    c_ = c
                self.index_plot.append(c_)
                c_ = c_[0] if isinstance(c_, tuple) else c_
                #cond_c_stats = self.stats_df[self.stats_df['Condition'] == cond].iloc[c_]
                #p = cond_c_stats[self.p_to_use]

                if not c_ in self.y_order:
                    continue
                hpos = self.y_order[c_]*len(self.conditions)+i*.8
                
                ax.boxplot(self.multi_df[cond].loc[c][~np.isnan(self.multi_df[cond].loc[c])], 
                            positions = [hpos],
                            widths=.4,
                            showfliers=self.showfliers,
                            patch_artist = True, boxprops={'facecolor':self.color_dict[c_]},medianprops={'color':'k'})

        # have to wait until all boxes are plot, to ensure alignment of annotation
        if self.stats_df is not None:
            for i, cond in enumerate(self.conditions):
                for j,c in enumerate(self.multi_df.index):
                    if isinstance(c, str):
                        c_ = eval(c)
                    else:
                        c_ = c
                    c_ = c_[0] if isinstance(c_, tuple) else c_
                    cond_c_stats = self.stats_df[self.stats_df['Condition'] == cond].iloc[c_]
                    p = cond_c_stats[self.p_to_use]
                    
                    if not c_ in self.y_order:
                        continue
                    hpos = self.y_order[c_]*len(self.conditions)+i*.8
                    
                    vpos = ax.get_ylim()[1]
                    if p < .05:
                        ax.text(hpos, vpos,'*'*sig_stars(p), ha='left', va='bottom', rotation=45) #TODO: sort out import
                    else:
                        ax.text(hpos, vpos,'n.s.', ha='left', va='bottom', rotation=45)
                    ax.text(hpos, vpos*1.15,'N='+str(cond_c_stats['N']),ha='center', va='bottom')
                    

        common = (find_common(*[s.split('_') for s in self.conditions]))  #TODO: sort out import
        uncommon = (find_uncommon(*[s.split('_') for s in self.conditions]))  #TODO: sort out import

        self.xlabels = np.repeat(uncommon,len({c: self.color_dict[c] for c in self.index_plot}))
        ax.set_xticklabels(self.xlabels, rotation=90)        
        ax.set_ylabel(self.y_label)
        
        ax.spines[['right', 'top']].set_visible(False)
        plt.title('_'.join(common),x=1.1,y=1.1)

        if self.showlegend:
            self.plot_legend()
        
        return fig
    
    def plot_legend(self):
        plt.legend(handles=[Patch(facecolor=self.color_dict[i]) for i in self.index_plot],
                   labels=[self.cluster_label[k] for k in self.index_plot],
                   loc='upper left',
                   bbox_to_anchor=(0, -0.5))
    
    def plot_groups(self, groups, normed=True):
        # TODO: should not depend on level(1) but rather last level, also if it is the only level
        # groups the multidf along axis 0, level 1 (states) and plots the resulting groups
        # group values are summed up
        if all([isinstance(l, str) for l in groups]):
            cluster_label_rev = {v: k for k,v in self.cluster_label.items()}
            grps_ = [[cluster_label_rev[l] for l in groups[i]] for i in range(len(groups))]
            grps_.append([l for l in self.cluster_label if not any([l in lst for lst in grps_]) and l!=-1 and l!= self.multi_df.index.get_level_values(0).unique()[0]]) #TODO l!=1 replace
        else:
            grps_ = groups
            grps_.append([l for l in self.multi_df.index.get_level_values(1).unique() if not any([l in lst for lst in grps_]) and l!=-1 and l!= self.multi_df.index.get_level_values(0).unique()[0]]) #TODO l!=1 replace

        preindex = pd.DataFrame(grps_).T.unstack().droplevel(1).dropna().astype(int).sort_values()
        multiindex = pd.MultiIndex.from_tuples(list(zip(preindex.index, preindex.values)))
        self.multi_df.index = multiindex
        self.multi_df = self.multi_df.groupby(level=0).sum()

        if normed:
            self.multi_df = self.multi_df/self.multi_df.sum(axis=0)
        fig = self.plot()
        return fig
        
    
def transplot(fr_transition_norm, cluster_labels, ordering=None, cmap='viridis', vmin=0, vmax=1, linked=None, label='[]'):
    fig = plt.figure()
    
    if ordering is None:
        ordering = list(range(len(fr_transition_norm)))
    
    if linked is not None:
        axs1 = fig.add_axes([0, .895, .2, .805])
        axs1.axis('off')
        dn = dendrogram(linked, orientation='left',ax= axs1, color_threshold=0,above_threshold_color='k')
        
    axs2 = fig.add_axes([0.35, .9, .8, .8])    
    im = axs2.imshow(fr_transition_norm.iloc[ordering,ordering], cmap =cmap, vmin=vmin, vmax=vmax) # x state i y state i+1
    axs2.set_xticks(range(len(ordering)))
    axs2.set_xticklabels([cluster_labels[k] for k in ordering], rotation=45,ha="center")
    axs2.set_xlabel('State i')
    axs2.set_yticks(range(len(ordering)))
    axs2.set_yticklabels([cluster_labels[k] for k in ordering])
    axs2.set_ylabel('State i+1')
    cbar = axs2.figure.colorbar(im, ax=axs2)
    #cbar.ax.set_ylabel("X^0.4 normalization", rotation=90, labelpad= 6)
    axs2.set_title(f"{label}")
    #axs2.set_title(f"transitions per sec\n{label}")
    return fig



class EthogramPlotter():
    def __init__(self, df, cluster_color, cluster_label, fps, plot_fps=None, xtick_spread = 30, multi_level=0):
        self.df = df
        self.fps = fps
        self.plot_fps = fps if plot_fps is None else plot_fps
        
        self.cluster_color = cluster_color
        self.cluster_label = cluster_label
        self.xtick_spread = xtick_spread
        
        self.init_color()

    def init_color(self):
        colors = [c for c in self.cluster_color.values()]
        self.cmap_cluster = mpl.colors.ListedColormap(colors, name='cluster', N=None)
        
    def stacked(self, data=None, ax=None, cbar = False, figsize=(4,5)):
        if ax is None:
            f, ax = plt.subplots(1, figsize=figsize)
        if data is None:
            data = self.df.copy()
        data = data[::int(self.fps/self.plot_fps)].T
        data = data.fillna(-1)
        #
        timeinsec = np.arange(data.shape[1]/self.plot_fps)
        # set limits .5 outside true range
        mat = ax.imshow(data, cmap=self.cmap_cluster, vmin=-1, 
                        vmax=5, aspect='auto', extent = (min(timeinsec), max(timeinsec),0,data.shape[0]), origin='lower', interpolation='nearest')
        #print(range(len(timeinsec))[::self.xtick_spread*self.fps])
        ax.set_xticks(np.arange(min(timeinsec), np.max(timeinsec), self.xtick_spread))
        return mat

    def multi_stack(self, adaptive_figsize=(4,5), xlim=(None,None), ylim=(None,None), multi_level=0):
        self.multi_level = multi_level
        self.conditions = self.df.columns.get_level_values(multi_level).unique()
        f, ax = plt.subplots(1,len(self.conditions), figsize=tuple(np.multiply(adaptive_figsize,(len(self.conditions),1))))
        if len(self.conditions) <= 1:
            ax = [ax]
        for i,cond in enumerate(self.conditions):
            self.stacked(self.df[cond], ax[i])
            ax[i].set_title(cond)
        ax[0].set_ylabel('Tracks')
        plt.setp(ax,xlim=xlim, ylim=ylim, xlabel= 'Time (s)')
        return f

    def single(self, y_column, metrics=[], smooth=30, adaptive_figsize=(20,2)):
        plot_col = self.df[y_column].copy().dropna()
        plot_col = plot_col[::int(self.fps/self.plot_fps)]
        onoff = proc.onoff_dict(plot_col, labels=np.unique(plot_col))
        timeinsec = np.arange(plot_col.shape[0]/self.plot_fps)
        fig, axs = plt.subplots(1+len(metrics), 1, 
                                figsize=tuple(np.multiply(adaptive_figsize,(1, 1+len(metrics)))),
                                constrained_layout=True)
        if not isinstance(axs,list):
            axs = [axs]
    
        for c in np.unique(plot_col).astype(int):
            axs[0].broken_barh(onoff[c],(0,1),facecolors = self.cluster_color[c])
        for i,met in enumerate(metrics):
            axs[i+1].plot(met.rolling(smooth, min_periods=0).mean(),c='k')
        for ax in axs:
            ax.set_xticks(np.arange(0, len(plot_col), self.xtick_spread*self.plot_fps))
            ax.set_xticklabels(np.arange(min(timeinsec), max(timeinsec), self.xtick_spread).astype(int))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(5*self.plot_fps))
        axs[-1].set_xlabel('sec')
        
        plt.legend(handles=[Patch(facecolor=self.cluster_color[i]) for i in np.unique(plot_col).astype(int)],
            labels=[self.cluster_label[k] for k in np.unique(plot_col)],
            ncol=3, loc='upper left',
            bbox_to_anchor=(0, -0.5))
        fig.suptitle(f'Ethogram of {y_column}',fontsize=16)
        return fig


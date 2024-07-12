import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import matplotlib as mpl
from matplotlib.patches import Patch

from . import process as proc

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
    def __init__(self, multi_df, stats_df, y_label, bonferroni, color_dict, multi_level = 0, plot_percentile=(0,100), showfliers=True, adaptive_figsize=True, figsize=(6,4)):
        self.multi_df = multi_df
        self.stats_df = stats_df
        self.plot_percentile = plot_percentile
        self.nan_percentiles()
        self.multi_level = multi_level
        self.bonferroni = bonferroni
        self.p_to_use = 'p' if not self.bonferroni else 'bonferroni p'
        self.y_label = y_label
        
        self.showfliers = showfliers
        self.adaptive_figsize = adaptive_figsize #True
        self.figsize = figsize #(6,4)
        self.color_dict = color_dict

        self.conditions = self.multi_df.columns.get_level_values(self.multi_level).unique()
        

    def nan_percentiles(self):
        self.percentile = (np.round(np.nanpercentile(self.multi_df,self.plot_percentile[0])).astype(int),np.round(np.nanpercentile(self.multi_df,self.plot_percentile[1])).astype(int))

    def plot(self):
        if self.adaptive_figsize:
            self.figsize = (self.figsize[0]*len(self.conditions), self.figsize[1])
                            
        fig, ax = plt.subplots(figsize=self.figsize)
            
        for i, cond in enumerate(self.conditions):
            for j,c in enumerate(self.multi_df.index):
                c_ = eval(c)
                c_ = c_[0] if isinstance(c_, tuple) else c_
                cond_c_stats = self.stats_df[self.stats_df['Condition'] == cond].iloc[c_]
                p = cond_c_stats[self.p_to_use]
                
                hpos = j*len(self.conditions)+i*.8
                
                ax.boxplot(self.multi_df[cond].loc[c][~np.isnan(self.multi_df[cond].loc[c])], 
                            positions = [hpos],
                            widths=.4,
                            showfliers=self.showfliers,
                            patch_artist = True, boxprops={'facecolor':self.color_dict[c_]},medianprops={'color':'k'})

        # have to wait until all boxes are plot, to ensure alignment of annotation
        for i, cond in enumerate(self.conditions):
            for j,c in enumerate(self.multi_df.index):
                c_ = eval(c)
                c_ = c_[0] if isinstance(c_, tuple) else c_
                cond_c_stats = self.stats_df[self.stats_df['Condition'] == cond].iloc[c_]
                p = cond_c_stats[self.p_to_use]
                
                hpos = j*len(self.conditions)+i*.8
                vpos = ax.get_ylim()[1]
                if p < .05:
                    ax.text(hpos, vpos,'*'*sig_stars(p), ha='left', va='bottom', rotation=45) #TODO: sort out import
                else:
                    ax.text(hpos, vpos,'n.s.', ha='left', va='bottom', rotation=45)
                ax.text(hpos, vpos*1.15,'N='+str(cond_c_stats['N']),ha='center', va='bottom')
                    

        common = (find_common(*[s.split('_') for s in self.conditions]))  #TODO: sort out import
        uncommon = (find_uncommon(*[s.split('_') for s in self.conditions]))  #TODO: sort out import

        self.xlabels = np.repeat(uncommon,len(self.color_dict)-1)
        ax.set_xticklabels(self.xlabels, rotation=90)        
        ax.set_ylabel(self.y_label)
        
        ax.spines[['right', 'top']].set_visible(False)
        plt.title('_'.join(common),x=1.1,y=1.1)
        
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
        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.25)
            # tell the colorbar to tick at integers
            n_clusters = len(self.cluster_label)
            offset = (n_clusters-1)/(n_clusters)
            tick_locs = np.arange(-0.6,n_clusters-2,offset)#*0.6#*(n_clusters-2)/(n_clusters-1)
            cbar = plt.colorbar(mat,cax=cax, ticks=tick_locs)#, norm=norm)#, shrink=0.5)
            cbar.ax.set_yticklabels([c for c in self.cluster_label.values()])
        return mat

    def multi_stack(self, adaptive_figsize=(4,5), xlim=(None,None), ylim=(None,None), multi_level=0):
        self.multi_level = multi_level
        self.conditions = self.df.columns.get_level_values(multi_level).unique()
        f, ax = plt.subplots(1,len(self.conditions), figsize=tuple(np.multiply(adaptive_figsize,(len(self.conditions),1))))
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
            axs[0].broken_barh(onoff[c],(0,1),facecolors = cluster_color[c])
        for i,met in enumerate(metrics):
            axs[i+1].plot(met.rolling(smooth, min_periods=0).mean(),c='k')
        for ax in axs:
            ax.set_xticks(np.arange(0, len(plot_col), self.xtick_spread*self.plot_fps))
            ax.set_xticklabels(np.arange(min(timeinsec), max(timeinsec), self.xtick_spread).astype(int))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(5*self.plot_fps))
        axs[-1].set_xlabel('sec')
        
        plt.legend(handles=[Patch(facecolor=cluster_color[i]) for i in np.unique(plot_col).astype(int)],
            labels=[self.cluster_label[k] for k in np.unique(plot_col)],
            ncol=3, loc='upper left',
            bbox_to_anchor=(0, -0.5))
        fig.suptitle(f'Ethogram of {y_column}',fontsize=16)
        return fig
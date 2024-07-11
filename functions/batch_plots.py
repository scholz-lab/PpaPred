import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

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
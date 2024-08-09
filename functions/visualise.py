import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import networkx as nx
from matplotlib.lines import Line2D


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

    
def transition_plotter(transition_toother, cluster_color, transition_self=None, figsize=(8,6), mut_scale=40, node_size=4000, 
                    other_connectionstyle = "arc3,rad=.15", self_connectionstyle="arc3,rad=0.5", node_alpha = 1, exclude_label = [-1], clu_group_label=None):
    if transition_self is None:
        print(transition_self)
        transition_self = transition_toother.copy().diagonal()
        np.fill_diagonal(transition_toother, 0)


        
    A = np.nan_to_num(np.around(transition_toother.T,3))
    if tuple(map(int, (nx.__version__.split(".")))) < tuple(map(int, ('3'.split(".")))) :
        G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    else:
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    
    weights = nx.get_edge_attributes(G,'weight').values()
    arr_out = [e[0] for e in G.edges(data=True)]
                        
    color_map = [cluster_color[k] for k in cluster_color if k not in exclude_label]
    edge_color = [cluster_color[c] for c in arr_out]
    #edge_alpha = [node_alpha[c] for c in arr_out]
                        
    fig, ax = plt.subplots(1, figsize=figsize)
    fig_w = fig.get_size_inches()[0]
    arrowsize = [w*mut_scale for w in weights]
                        
    if clu_group_label is None:
        labels = dict(zip(range(len(G)),range(len(G))))
    else:
        labels = dict(zip(range(len(G)),  [clu_group_label[k] for k in clu_group_label if k != -1]))
        
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
        (A, _, C, D) =  SemiCirc_coordinates(xy_s, xy_t, r=0.2)
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

def ethogram_plotter(d, y, onoff,  smooth, cluster_color, figsize=(20,5), fps=30,xtick_spread=100, d_toplot=['velocity', 'rate'], d_bar_alpha =0.3):
    timeinsec = np.array(range(len(d)))/fps
    
    fig, axs = plt.subplots(3,1, figsize=figsize,constrained_layout=True)
    
    for c in np.unique(y).astype(int):
        axs[0].broken_barh(onoff[c],(0,1),facecolors = cluster_color[c])
    axs[0].set_xticks(range(len(timeinsec))[::xtick_spread*fps])
    axs[0].set_xticklabels(timeinsec[::xtick_spread*fps].astype(int))
    axs[0].set_title(f'Cluster preditcion (smoothed {smooth/fps} sec).')
    for i,c in enumerate(d_toplot):
        for c_ in np.unique(y).astype(int):
            #axs[i+1].broken_barh(onoff[c_],(min(d[c]),max(d[c])),facecolors = cluster_color[c_], alpha=0.6, zorder=0)
            axs[i+1].broken_barh(onoff[c_],(np.nanmin(d[c]),np.nanmax(d[c])-np.nanmin(d[c])),facecolors = cluster_color[c_], alpha=d_bar_alpha, zorder=0)
        axs[i+1].plot(d[c].rolling(30, min_periods=0).mean(),c='k')
        axs[i+1].set_xticks(range(len(timeinsec))[::100*30])
        axs[i+1].set_xticklabels(timeinsec[::100*30].astype(int))
        axs[i+1].set_title(f"{c} (smoothed, 1 sec)")
    axs[2].set_xlabel('sec')

    plt.legend(handles=[Patch(facecolor=cluster_color[i]) for i in np.unique(y).astype(int)],
          labels=[clu_group_label[k] for k in cluster_label if k in np.unique(y)],
          ncol=3, loc='upper left',
          bbox_to_anchor=(0, -0.5))
    fig.suptitle(f'Ethogram of {fn}',fontsize=16)
    return fig
    


def ethogram_only_plotter(y, onoff, cluster_color, figsize=(20,2), fps=30,xtick_spread=100):
    timeinsec = np.array(range(len(y)))/fps
    
    fig, ax = plt.subplots(1, figsize=figsize,constrained_layout=True)
    
    for c in np.unique(y).astype(int):
        axs[0].broken_barh(onoff[c],(0,1),facecolors = cluster_color[c])
    axs[0].set_xticks(range(len(timeinsec))[::xtick_spread*fps])
    axs[0].set_xticklabels(timeinsec[::xtick_spread*fps].astype(int))
    axs[0].set_title(f'Cluster preditcion (smoothed {smooth/fps} sec).')

    plt.legend(handles=[Patch(facecolor=cluster_color[i]) for i in np.unique(y).astype(int)],
          labels=[clu_group_label[k] for k in cluster_label if k in np.unique(y)],
          ncol=3, loc='upper left',
          bbox_to_anchor=(0, -0.5))
    fig.suptitle(f'Ethogram of {fn}',fontsize=16)
    return fig

def CLtrajectory_plotter(CLine, XY, cluster_color=['k'], cluster_label=[0], y = None, fn='', figsize=(10,10)):
    fig, ax = plt.subplots(figsize=figsize)
    
    adjustCL = (CLine-np.nanmean(CLine))+np.repeat(XY.reshape(XY.shape[0],1,XY.shape[1]), CLine.shape[1], axis=1)# fits better than subtracting 50
    adjustXY = XY-np.nanmean(XY, axis=0)
    if y is not None:
        legend_elements = [Line2D([0], [0],color=cluster_color[i], label=cluster_label [i]) for i in cluster_label]
        for l in np.unique(y).astype(int):
            il = np.where(y == l)[0]
            ax.plot(*adjustCL[il].T, c=cluster_color[l], alpha = 0.1)
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1,1))
    else:
        ax.plot(*adjustCL.T, c=cluster_color[0], alpha = 0.1)
    ax.set_title(fn)
    ax.axis('equal')
    return fig
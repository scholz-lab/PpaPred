import numpy as np
import pandas as pd
import json
import os
import tqdm
import itertools
import scipy
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram

# local imports
from . import io
from . import read_write as rw

def append_df_to_multi(dict_, level_zero, metric_multi = None, drop_index=[-1]):
    """
    Creates a multiindex DataFrame, from dictionary
    """
    try:
        df = pd.DataFrame(dict_).droplevel(0, axis=1)
    except:
        df = pd.DataFrame([])
        for entry in dict_:
            seri = pd.Series(dict_[entry], name=entry[1])
            df = pd.concat([df, seri], axis=1)
    
    df.columns = pd.MultiIndex.from_product([[level_zero], df.columns])
    
    for idx in drop_index:
        if idx or str(idx) in df.index:
            try:
                df = df.drop(idx)
            except:
                try:
                    df = df.drop(str(idx))
                except:
                    pass
                    
    if metric_multi is None:
        metric_multi = df
    else:
        metric_multi = pd.concat([metric_multi, df], axis=1)

    return metric_multi

class BatchCondition():
    """
    Creates and loads batch json files. That contain releveant information for state predictions.
    """
    def __init__(self, inpath, data_str, jsonpath=None):
        self.inpath = inpath
        self.jsonpath = jsonpath
        self.data_str = data_str
        self.suffix = 'batch'
        self.json_path()

    def json_path(self):
        if self.jsonpath is None:
            self.jsonpath = self.inpath
        self.jsonpath = os.path.join(self.jsonpath,f'{self.data_str}_{self.suffix}.json')

    def create_json(self, append = False, overwrite = False):
        if self.check_json_exists() and not overwrite and not append:
            return self.load_json()
        loc_all, loc_summ, loc_trans = io.walkdir_filter(self.inpath, self.data_str, specific_patterns=['prediction.json', 'summary.csv','transitions.csv'])
        for fn,fpath in tqdm.tqdm(loc_all.items()):
            id = '_'.join(fn.split('_')[:-1])
            data = pd.read_json(fpath, orient='split')
            y = data['prediction']
            proba = data.filter(regex='proba')
            proba_idx = proba.columns.str.split('_', expand=True)
            proba.columns = proba_idx
            mean_probas = {cl:np.nanmean(proba.loc[:,('proba',cl)][y == eval(cl)]) for cl in proba.columns.levels[1]}
            
            summ_ = pd.read_csv([l for l in loc_summ.values() if id in l][0])

            fr_transition_ = pd.read_csv([l for l in loc_trans.values() if id in l][0], index_col=0)
            fr_transition_[fr_transition_==0] = np.nan # for now until processing in FeedingPrediction is fixed
            fr_transition_tuple = dict(zip(str(list(itertools.product(fr_transition_.columns.astype(int), fr_transition_.index))).strip('[()]').split('), ('), fr_transition_.values.T.flatten()))
    
            data_mean = data[['velocity', 'rate', 'prediction']].groupby('prediction').mean().reindex(range(-1,8))
            
            # prep of json file structure
            etho = {id:{'count':summ_.duration_count.fillna(0).to_dict(),
                        'mean duration':summ_.duration_mean.to_dict(),
                        'rel time in': summ_.duration_relative.fillna(0).to_dict(),
                        'mean velocity': data_mean.velocity.to_dict(),
                        'mean rate': data_mean.rate.to_dict(),
                        'mean transitions':fr_transition_tuple,
                        'mean prediction probability': mean_probas,
                        'ethogram':y.to_list()}}

            # if file exists and overwrite is false
            ow_org = append
            if os.path.isfile(self.jsonpath) and append:
                with open(self.jsonpath, "r") as jsonfile:
                    batch = json.load(jsonfile)
            else:
                batch = {}
                append = True
            
            batch.update(etho)
            jsnF = json.dumps(batch, indent = 4, cls=rw.NanConverter)
            with open(self.jsonpath, "w") as outfile:
                outfile.write(jsnF)
        with open(self.jsonpath, 'r') as f:
            self.batch = json.load(f)
        append = ow_org

    def check_json_exists(self):
        return os.path.exists(self.jsonpath)

    def load_json(self):
        if not self.check_json_exists():
            print('Json file not found. Json will be created first.')
            return self.create_json()
        with open(self.jsonpath, 'r') as f:
            self.batch = json.load(f)
    
    def load_data_from_keys(self, key):
        def traverse_dict(d, key):
            if isinstance(d, dict):
                if key in d.keys():
                    return {key: d[key]}
                else:
                    return {k: traverse_dict(v, key) for k, v in d.items()}
            elif isinstance(d, list):
                return [traverse_dict(x, key) for x in d]
            else:
                return d

        nested_dict = traverse_dict(self.batch, key)
        return {(innerKey, outerKey): values for outerKey, innerDict in nested_dict.items() for innerKey, values in innerDict.items()}
    

class MannWhitneyU_frommultidf():
    def __init__(self, multi_df, exp, control, bonferroni, N_bonf_tests, multi_level=0):
        self.multi_df = multi_df
        self.exp = exp
        self.control = control
        self.bonferroni = bonferroni
        self.N_bonf_tests = N_bonf_tests
        self.multi_level = multi_level
        self.conditions = self.multi_df.columns.get_level_values(self.multi_level).unique()
        self.init_csv()

    def init_csv(self):
        self.stats_csv = pd.DataFrame([], columns = ['Experiment', 'Population', 'Condition', 
                                                     'State', 'N', 
                                                     'Mean Rank', 'Sum Rank', 
                                                     'U1', 'U2', 'Mann-Whitney U', 
                                                     'p', 'number of tests', 'bonferroni p'])
    def test(self, cond):
        U1s, ps = stats.mannwhitneyu(self.multi_df[self.control], self.multi_df[cond], axis=1, nan_policy='omit')
        ps_adjusted = ps*self.N_bonf_tests
        
        cond_n = np.count_nonzero(~np.isnan(self.multi_df[cond]), axis=1)
        pop_n = np.count_nonzero(~np.isnan(self.multi_df[self.control]), axis=1)

        U2s = pop_n*cond_n - U1s
        Us = np.min((U1s, U2s), axis=0)

        rank_multi = pd.concat([self.multi_df[self.control], self.multi_df[cond]], axis=1,
                           keys={'pop':self.multi_df[self.control].columns, 'cond':self.multi_df[cond].columns,})
        cond_idx, = np.where(rank_multi.columns.get_level_values(0)=='cond')
        ranks = stats.rankdata(rank_multi, axis=1, nan_policy='omit')
        cond_ranks = ranks[:,cond_idx]
        cond_ranks_mean = np.nanmean(cond_ranks, axis=1)
        cond_ranks_sum = np.nansum(cond_ranks, axis=1)

        # create dict of current condition
        stats_cond = {'Experiment': [self.exp]*self.multi_df.shape[0], 
                      'Population': [self.control]*self.multi_df.shape[0], 
                      'Condition': [cond]*self.multi_df.shape[0], 
                      'State': self.multi_df.index, 
                      'N': cond_n, 
                      'Mean Rank': cond_ranks_mean, 
                      'Sum Rank': cond_ranks_sum, 
                      'U1': U1s, 
                      'U2': U2s, 
                      'Mann-Whitney U': Us, 
                      'p': ps, 
                      'number of tests': [self.N_bonf_tests]*self.multi_df.shape[0], 
                      'bonferroni p': ps_adjusted}
        # concat stats dict to dataframe stats_csv
        self.stats_csv = pd.concat([self.stats_csv, pd.DataFrame(stats_cond)])
        
        return self.stats_csv

    def iterate_conditions(self):
        for cond in self.conditions:
            _ = self.test(cond)
        return self.stats_csv


class BatchTransitions_frommultidf():
    def __init__(self, multi_df, control=None, with_self=False, norm_over='out', Z=None, drop_states=[-1], multi_level=0):
        # multi df
        self.multi_df = multi_df.copy()
        self.control = control
        self.multi_level = multi_level
        self.conditions = self.multi_df.columns.get_level_values(self.multi_level).unique()
        # matrix
        self.with_self = with_self
        self.norm_method = self.get_norm_method(norm_over)
        self.drop_states = drop_states
        # linkage
        self.Z = Z
        # init
        self.init_index()
        self.get_control()

    def init_index(self):
        self.multi_df.index = pd.MultiIndex.from_tuples([eval(i) for i in self.multi_df.index])
        for drop in self.drop_states:
            self.multi_df = self.multi_df.drop(drop, level = 0)
            self.multi_df = self.multi_df.drop(drop, level = 1)

    def get_control(self):
        if self.control is None:
            self.control = self.conditions[0]

    def get_norm_method(self, norm_over):
        if norm_over == 'out':
            return self.norm_out
        if norm_over == 'in':
            return self.norm_in
        if norm_over == 'inout':
            return self.norm_inout

    def norm_out(self, transitions):
        return transitions/transitions.sum(axis=0)
    
    def norm_in(self, transitions):
        return (transitions.T/transitions.sum(axis=1)).T
    
    def norm_inout(self, transitions):
        return pd.DataFrame(np.triu(transitions)/np.sum(np.triu(transitions))+np.tril(transitions)/np.sum(np.tril(transitions)))

    def norm_symsum(self, transitions):
        # to symmetrize the matrix the upper and lower triangle are translated and added to the original matrix
        symsum = ((transitions+np.triu(transitions).T+np.tril(transitions).T)) #TODO should be without transitions+
        return symsum/sum(symsum)

    def normalize(self, cond_df, method = None):
        if method is None:
            method = self.norm_method
        self.transitions = cond_df.mean(axis=1).unstack(level=0).fillna(0)
        if not self.with_self:
            self.transitions_self = np.diagonal(self.transitions.values).copy()
            np.fill_diagonal(self.transitions.values, 0)
        self.transitions_norm = method(self.transitions)
        return self.transitions_norm

    def normalize_multi(self, level=0):
        self.conditions = self.multi_df.columns.get_level_values(level).unique()
        self.transitions_norm_all = {}
        for cond in self.conditions:
            self.transitions_norm_all[cond] = self.normalize(self.multi_df[cond])
        return self.transitions_norm_all
        
    def linkage(self, method='single', metric='euclidean', optimal_ordering=True):
        # perform hierarchical linkage based on control
        self.transitions_symsum = self.normalize(self.multi_df[self.control], method=self.norm_symsum)
        self.transitions_symsum[self.transitions_symsum == 0] = 1e-10
        self.dist_mat = 1/self.transitions_symsum
        np.fill_diagonal(self.dist_mat.values, 0)
        self.pairwise_dist = scipy.spatial.distance.squareform(self.dist_mat)
        self.Z = linkage(self.pairwise_dist, method, metric=metric, optimal_ordering=optimal_ordering)
        return self.Z

    def dendrogram(self, **kwargs):
        if self.Z is None:
            self.linkage()
        self.dendrogram = dendrogram(self.Z, **kwargs)
        return self.dendrogram
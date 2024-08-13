import json
import numpy as np
import pandas as pd

def load_data_from_keys(json_file, key):
    """
    Load data from a nested json file. Json file is opened and iteratively searched for specified key in data.
    A dictionary is returned containing all matching entries.
    Args:
        json_file (str): path to json file
        key (str): name of key that should be read.
    Returns:
        dict with {(a_key_n0,...,a_key_ni): a_value, (b_key_n0,...,b_key_ni): b_value, ...}
    """
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

    with open(json_file, 'r') as f:
        data = json.load(f)
    nested_dict = traverse_dict(data, key)
    return {(innerKey, outerKey): values for outerKey, innerDict in nested_dict.items() for innerKey, values in innerDict.items()}

# used
class NpIntEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

#used
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
    
# used
def iterable_exps(config_dict, settings_key='Settings', run_key='run', exp_key='Experiments', cond_key='conditions', control_key='control'):
    run_exps = config_dict[settings_key][run_key]
    if run_exps == 'All' or run_exps == 'all':
        run_exps = list(config_dict[exp_key].keys())
    elif isinstance(run_exps, str):
        run_exps = [run_exps]
    exps_include = []
    exps_statpop = []
    for exp in run_exps:
        include = config_dict[exp_key][exp][cond_key]
        if control_key in config_dict[exp_key][exp]:
            statpop = config_dict[exp_key][exp][control_key]
        else:
            statpop = include[0]
        
        if statpop not in include:
            include.insert(0, statpop)
        exps_include.append(include)
        exps_statpop.append(statpop)
    return run_exps, exps_include, exps_statpop

def load_tolist(loc, r = False, label=None, labelstonum=None, droplabels = None, dropcolumns = None, target2 = None, labelstonum2=None, droplabelcol=True):
    # read data from json files
    # create empty array
    data_lst = []
    target_lst = []
    target2_lst = []
    if isinstance(loc, str):
        fdata = pd.read_json(loc, orient='split')
        if target2 is not None:
            if labelstonum2 is not None:
                target2_lst.append(fdata[target2].replace(labelstonum2.keys(), labelstonum2.values()))
            else:
                target2_lst.append(fdata[target2])
            
        if dropcolumns is not None:
            fdata = fdata.drop(dropcolumns, axis=1)
            
        if droplabels is not None:
                for drops in droplabels:
                    fdata.iloc[np.where(fdata[label] == drops)[0]] = np.nan
                    
        if labelstonum is not None:
            target_lst.append(fdata[label].replace(labelstonum.keys(), labelstonum.values()))
        if droplabelcol:
            target_lst.append(fdata[label])
            data_lst.append(fdata.drop(label, axis=1))
        else:
            data_lst.append(fdata)
    else:
        for i,fn in enumerate(loc):
            print(fn)
            fdata = pd.read_json(loc[fn], orient='split')
            if target2 is not None:
                if labelstonum2 is not None:
                    target2_lst.append(fdata[target2].replace(labelstonum2.keys(), labelstonum2.values()))
                else:
                    target2_lst.append(fdata[target2])
                
            if dropcolumns is not None:
                fdata = fdata.drop(dropcolumns, axis=1)
                
            if droplabels is not None:
                for drops in droplabels:
                    fdata.iloc[np.where(fdata[label] == drops)[0]] = np.nan

            if labelstonum is not None:
                target_lst.append(fdata[label].replace(labelstonum.keys(), labelstonum.values()))
            
            if droplabelcol:
                target_lst.append(fdata[label])
                data_lst.append(fdata.drop(label, axis=1))
            else:
                data_lst.append(fdata)
    if target2 is not None:
        return data_lst, target_lst, target2_lst
    elif label is not None:
        return data_lst, target_lst
    return data_lst
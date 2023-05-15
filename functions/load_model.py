import numpy as np
import pandas as pd


def load_tolist(loc, label=None, labelstonum=None, droplabels = None, dropcolumns = None, target2 = None, labelstonum2=None, droplabelcol=True):
    # read data from json files
    # create empty array
    data_lst = []
    target_lst = []
    target2_lst = []
    if isinstance(loc, str):
        fdata = pd.read_json(loc, orient='split')
        if target2 is not None:
            if labestonum2 is not None:
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
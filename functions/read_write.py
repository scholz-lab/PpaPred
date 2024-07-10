import json
import numpy as np

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
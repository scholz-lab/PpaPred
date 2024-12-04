import logging
import os
import sys

# used
def setup_logger(name, filename, level=logging.INFO, logformat ='%(asctime)s %(levelname)s %(message)s'):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    handler = logging.FileHandler(filename)        
    handler.setFormatter(logging.Formatter(logformat))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

# used
def makedir(dirname, basepath = ''):
    dirpath = os.path.join(basepath, dirname)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    return dirpath

# used
def walkdir_filter(inpath, data_str, specific_patterns=None, fileendings=['csv','json'],):
    pattern_dir = [os.path.join(root, name) for root, dirs, files in os.walk(inpath) for name in dirs if data_str in name]
    #all_files = [os.path.join(root, name) for root, dirs, files in os.walk(pattern_dir) for name in files if any([name.endswith(suff) for suff in fileendings])]
    all_files = [[os.path.join(root, name) for root, dirs, files in os.walk(pat_dir) for name in files if any([name.endswith(suff) for suff in fileendings])] for pat_dir in pattern_dir]
    all_files = [v for lst in all_files for v in lst]
    list_return = []
    if specific_patterns is None:
        return all_files
    for pat in specific_patterns:
        if isinstance(pat,str) and pat != '':
            list_return.append({os.path.basename(f):f for f in all_files if pat in f})
        else:
            list_return.append({})
    return list_return

def listdir_filter(inpath, data_str, specific_patterns=None, fileendings=['csv','json'],):
    all_files = [os.path.join(inpath, file) for file in os.listdir(inpath) if data_str in file]
    list_return = []
    if specific_patterns is None:
        return all_files
    for pat in specific_patterns:
        if isinstance(pat,str) and pat != '':
            list_return.append({os.path.basename(f):f for f in all_files if pat in f})
        else:
            list_return.append({})
    return list_return

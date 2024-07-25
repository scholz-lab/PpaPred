import logging
import os
import re
import inspect
import sys

def setup_logger(name, filename, level=logging.INFO, logformat ='%(asctime)s %(levelname)s %(message)s'):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    handler = logging.FileHandler(filename)        
    handler.setFormatter(logging.Formatter(logformat))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

def makedir(dirname, basepath = ''):
    dirpath = os.path.join(basepath, dirname)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    return dirpath

def str_var(args):
    frame = inspect.currentframe()
    frame = inspect.getouterframes(frame)[1]
    string = inspect.getframeinfo(frame[0]).code_context[0].strip()
    args = re.split(', |]',string[string.find('[') + 1:-1])
    names = []
    for i in args:
        if i.find('=') != -1:
            names.append(i.split('=')[1].strip())
        elif i == '':
            pass
        else:
            names.append(i)    
    return (names)

def walkdir_filter(inpath, data_str, specific_patterns=None, fileendings=['csv','json']):
    pattern_dir = [os.path.join(root, name) for root, dirs, files in os.walk(inpath) for name in dirs if data_str in name]
    #all_files = [os.path.join(root, name) for root, dirs, files in os.walk(pattern_dir) for name in files if any([name.endswith(suff) for suff in fileendings])]
    all_files = [[os.path.join(root, name) for root, dirs, files in os.walk(pat_dir) for name in files if any([name.endswith(suff) for suff in fileendings])] for pat_dir in pattern_dir]
    all_files = [v for lst in all_files for v in lst]
    
    list_return = []
    if specific_patterns is None:
        return all_files
    for pat in specific_patterns:
        list_return.append({os.path.basename(f):f for f in all_files if pat in f})
    return list_return

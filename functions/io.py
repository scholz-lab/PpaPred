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
        os.mkdir(dirpath)
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
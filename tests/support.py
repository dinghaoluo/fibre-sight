'''
Created on 24 July 2026

shared paths and subprocess environment for standalone tests

@author: Dinghao Luo
'''

#%% imports
from pathlib import Path
import os
import sys


#%% paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'


#%% helpers
def add_source_to_path():
    '''put the checkout source tree first on sys.path'''
    source_path = str(SRC_ROOT)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)


def source_environment():
    '''return an environment which imports the checkout source tree'''
    env = os.environ.copy()
    existing = env.get('PYTHONPATH', '')
    paths = [str(SRC_ROOT)]
    if existing:
        paths.append(existing)
    env['PYTHONPATH'] = os.pathsep.join(paths)
    return env

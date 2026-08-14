'''
Created on 6 April 2026

Modified on 24 July 2026 to resolve recipes against the local workspace
Modified on 14 August 2026

read and write the YAML training recipes

@author: Dinghao Luo
'''

#%% imports
from pathlib import Path

import yaml


#%% io
def load_recipe(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_recipe(recipe, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(recipe, f, sort_keys=False)


def resolve_path(path, base_root):
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(base_root) / path

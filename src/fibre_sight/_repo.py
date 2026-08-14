'''
Created on 6 April 2026

Modified on 23 June 2026
Modified on 24 July 2026 to replace lab paths with a local workspace
Modified on 14 August 2026

paths used by the command-line tools and GUI

@author: Dinghao Luo
'''

#%% imports
from pathlib import Path
import os


#%% package and workspace paths
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]

if 'FIBRE_SIGHT_WORKSPACE' in os.environ:
    WORKSPACE_ROOT = Path(os.environ['FIBRE_SIGHT_WORKSPACE']).expanduser()
elif (REPO_ROOT / 'pyproject.toml').is_file():
    WORKSPACE_ROOT = REPO_ROOT / 'workspace'
else:
    WORKSPACE_ROOT = Path.home() / 'fibre-sight'

SOURCE_ROOT = WORKSPACE_ROOT / 'labelled_sessions'
OUTPUT_ROOT = WORKSPACE_ROOT / 'output'
FIGURE_ROOT = OUTPUT_ROOT / 'figures'

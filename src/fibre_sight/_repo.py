'''
Created on 6 April 2026

Modified on 23 June 2026
Modified on 24 July 2026 to replace lab paths with a local workspace
paths used by the command-line tools and workbench

@author: Dinghao Luo
'''

#%% imports
from pathlib import Path
import os


#%% package and workspace paths
def package_root():
    return Path(__file__).resolve().parent


def package_path(*parts):
    return package_root().joinpath(*parts)


def _clone_root():
    package_dir = package_root()
    for candidate in list(package_dir.parents)[:3]:
        source_package = candidate / 'src' / 'fibre_sight'
        if (candidate / 'pyproject.toml').is_file() and source_package.resolve() == package_dir:
            return candidate
    return None


def get_workspace_root():
    configured = os.environ.get('FIBRE_SIGHT_WORKSPACE', '').strip()
    if configured:
        return Path(configured).expanduser()

    clone_root = _clone_root()
    if clone_root is not None:
        return clone_root / 'workspace'
    return Path.home() / 'fibre-sight'


def default_source_root():
    return get_workspace_root() / 'labelled_sessions'


def default_output_root():
    return get_workspace_root() / 'output'


def default_figure_root():
    return default_output_root() / 'figures'

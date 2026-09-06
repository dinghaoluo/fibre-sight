'''
Created on 24 July 2026

Modified on 14 August 2026
Modified on 20 August 2026

matplotlib formatting used by the analysis figures

@author: Dinghao Luo
'''

#%% imports
from ._repo import PACKAGE_ROOT


#%% plotting
_FONT_READY = False


def mpl_formatting():
    global _FONT_READY
    import matplotlib
    from matplotlib import font_manager

    if not _FONT_READY:
        font_root = PACKAGE_ROOT / 'assets' / 'fonts' / 'mononoki'
        for font_path in sorted(font_root.glob('*.ttf')):
            font_manager.fontManager.addfont(str(font_path))
        _FONT_READY = True

    matplotlib.rcParams.update(
        {
            'font.family': 'mononoki',
            'font.monospace': ['mononoki'],
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'savefig.facecolor': 'white',
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
        }
    )

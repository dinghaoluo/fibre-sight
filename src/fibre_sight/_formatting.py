'''
Created on 24 July 2026

Modified on 14 August 2026

matplotlib formatting used by the analysis figures

@author: Dinghao Luo
'''

#%% plotting
def mpl_formatting():
    import matplotlib

    matplotlib.rcParams.update(
        {
            'font.family': 'Arial',
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
        }
    )

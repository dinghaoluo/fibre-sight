'''
Created on 24 July 2026

small console and matplotlib formatting helpers

@author: Dinghao Luo
'''

#%% console
def print_files_saved(entries, gap=1):
    gap = min(gap, 1)
    prefix = '\n' * gap
    bar = '-' * 12
    print(f'{prefix}{bar}\nFILES SAVED\n{bar}')
    for label, path in entries:
        print(f'{label}: {path}')


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

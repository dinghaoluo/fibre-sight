'''
Created on 25 July 2026

Modified on 14 August 2026

select the available PyTorch compute device

@author: Dinghao Luo
'''

#%% selection
def get_device(device='auto'):
    import torch

    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    return torch.device(device)

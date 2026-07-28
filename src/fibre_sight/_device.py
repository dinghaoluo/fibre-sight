'''
Created on 25 July 2026

select the available PyTorch compute device

@author: Dinghao Luo
'''

#%% selection
def automatic_device_name():
    import torch

    if torch.cuda.is_available():
        return 'cuda'

    if torch.backends.mps.is_available():
        return 'mps'

    return 'cpu'


def resolve_device(device='auto'):
    import torch

    if isinstance(device, torch.device):
        resolved = device
    else:
        device_name = str(device).strip().lower()
        if device_name == 'auto':
            device_name = automatic_device_name()
        resolved = torch.device(device_name)

    if resolved.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested, but no CUDA device is available')

    if resolved.type == 'mps':
        # an x86_64 or non-macOS build carries the module but never builds the backend
        if not torch.backends.mps.is_built():
            raise RuntimeError('MPS was requested, but this PyTorch build has no MPS support')
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS was requested, but it is not available on this Mac')

    return resolved

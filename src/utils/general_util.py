import pickle
import random
import torch
import numpy as np


def getCurrentMemoryUsage():
    # From https://stackoverflow.com/questions/938733/total-memory-used-by-python-process
    ''' Memory usage in kB '''

    with open('/proc/self/status') as f:
        memusage = f.read().split('VmRSS:')[1].split('\n')[0][:-3]

    return int(memusage.strip())

def save_pickle(data, fname):
    '''Save data to pickle file'''
    with open(f'{fname}.pkl', 'wb') as f:
        pickle.dump(data, f, protocol=4)


def load_pickle(fname):
    '''Load data from pickle file'''
    try:
        with open(f'{fname}.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("Wrong file path or file does not exist!")


def set_seed(seed=None, seed_torch=True):
    # Adapated from NMA-DL tutorials (https://deeplearning.neuromatch.io/)

    if seed is None:
        seed = np.random.choice(2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    print(f'Random seed {seed} has been set.')


def seed_worker(worker_id):
    # In case that `DataLoader` is used
    # Adapated from NMA-DL tutorials (https://deeplearning.neuromatch.io/)

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    

def set_device():
    # Adapated from NMA-DL tutorials (https://deeplearning.neuromatch.io/)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("GPU is not enabled in this notebook.")
    else:
        print("GPU is enabled in this notebook.")
    return device


def map_list_with_dict(ls, d):
    '''Map list as key to dictionery values'''
    return list(map(d.get, ls))


def sample_idx(data, n_sample=1, seed=0):
    '''
        Randomly select samples from data, returns index
        Example usage: data[sample_idx(data)]
    '''
    set_seed(seed=seed)
    return np.random.choice(len(data), size=n_sample, replace=False)
    
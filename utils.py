def set_seed(seed):
    import numpy as np
    import random
    import torch
    
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multi-gpu
    torch.manual_seed(seed)

def nowdt():
    """
    get string representation of date and time of now()
    """
    from datetime import datetime

    now = datetime.now()
    return now.strftime("%d/%m/%Y %H:%M:%S")


def mymkdir(dir_name, clean=False):
    import os
    import shutil

    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    elif clean:
        # rmtree(train_dir)
        # print('To remove ', train_dir)
        yn = input('Delete directory %s, y or n? ' % dir_name)
        if yn.lower() == 'y':
            shutil.rmtree(dir_name)
            print('Cleaning and recreating %s ...' % dir_name)
            os.makedirs(dir_name)


def list_multi_del(lst:list, indices):
    """
    delete elements from list by indices
    """
    indices = list(set(indices))
    for idx in sorted(indices, reverse=True):
        del lst[idx]


def row_cos_sim(X, Y):
    """
    calcuate 
    X: [n, d] Tensor
    Y: [n, d] Tensor
    ret: [n] Tensor
    """
    import torch
    import torch.nn.functional as F
    norm_X, norm_Y = F.normalize(X), F.normalize(Y)
    return torch.mul(norm_X, norm_Y).sum(1)


def torch_the_same(X, Y, eps=1e-8):
    """
    return whether two Tensors are the same numerically
    useful for unit test and debug
    """
    return (X - Y).abs().min() < eps

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


class EarlyStopping(object):
    def __init__(self, patience, save_dir):
        self.patience = patience
        self.counter = 0
        self.best_metrics = None
        self.save_dir = save_dir
        self.early_stop = False

    def step(self, model, epoch, *metrics):
        if self.best_metrics is None:
            self.best_metrics = [metric for metric in metrics]
            self.save_checkpoint(model, epoch)
        elif all([metric < best_metric for metric, best_metric in zip(metrics, self.best_metrics)]):
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if all([metric > best_metric for metric, best_metric in zip(metrics, self.best_metrics)]):
                self.save_checkpoint(model, epoch)
            for i, (metric, best_metric) in enumerate(zip(metrics, self.best_metrics)):
                self.best_metrics[i] = max(metric, best_metric)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model, epoch):
        save_path = os.path.join(self.save_dir, '%d.pth'%epoch)
        torch.save(model.state_dict(), save_path)

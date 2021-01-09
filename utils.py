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
    torch.backends.cudnn.deterministic = True

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


class EarlyStopping:
    def __init__(self, patience=10, path=''):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_path = path

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        save_dir = '/'.join(self.save_path.split('/')[:-1])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(model.state_dict(), self.save_path)
        
 
class Logger(object):
    """
    这个类的目的是尽可能不改变原始代码的情况下, 使得程序的输出同时打印在控制台和保存在文件中
    用法: 只需在程序中加入一行 `sys.stdout = Logger(log_file_path)` 即可
    """
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

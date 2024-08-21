import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import os
import numpy as np

def get_free_gpu():
#    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
    memory_used = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    print("Memory used: ", memory_used)
    return np.argmin(memory_used)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True

if __name__ == '__main__':
    free_gpu_id = get_free_gpu()
    print(free_gpu_id)

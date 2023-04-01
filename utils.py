import logging
import logging.handlers
import os
import sys
import time
from tqdm.autonotebook import tqdm
import torch.distributed as dist
import numpy as np
import scipy.sparse as sp
import itertools
from torch.utils.data import Dataset

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def all_gather(item, world_size, filter_none=True):
    item_list = [None for _ in range(world_size)]
    if world_size > 1:
        dist.all_gather_object(item_list, item, group=dist.group.WORLD)
    else:
        item_list = [item]
    if filter_none:
        return [x for x in item_list if x is not None]
    else:
        return item_list


def all_sync(item, world_size, source=0):
    return all_gather(item, world_size)[source]


def all_gather_reduce(item, world_size, reduction='sum'):
    if reduction == 'sum':
        return sum(all_gather(item, world_size))
    elif reduction == 'mean':
        return sum(all_gather(item, world_size))/world_size
    elif reduction == 'npvstack':
        return np.vstack(all_gather(item, world_size))
    elif reduction == 'npconcat':
        return np.concatenate(all_gather(item, world_size))
    elif reduction == 'spvstack':
        return sp.vstack(all_gather(item, world_size))
    elif reduction == 'listconcat':
        return list(itertools.chain(*all_gather(item, world_size)))
    elif reduction == 'shape':
        items_list = all_gather(item, world_size)
        print(item, items_list, world_size)
        return (sum([x[0] for x in items_list]), sum([x[1] for x in items_list])//world_size)
    elif reduction == 'weighted_mean':
        weights = all_gather(item[0], world_size)
        items_list = all_gather(item[1], world_size)
        return sum([x*y for x, y in zip(weights, items_list)]) / sum(weights)
    else:
        assert False, f'unknown reduction {reduction}'


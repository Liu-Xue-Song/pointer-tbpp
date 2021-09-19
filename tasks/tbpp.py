"""Defines the main task for the TSP

The TSP is defined by the following traits:
    1. Each city in the list must be visited once and only once
    2. The salesman must return to the original node at the end of the tour

Since the TSP doesn't have dynamic elements, we return an empty list on
__getitem__, which gets processed in trainer.py to be None

"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class TBPPDataset(Dataset):

    def __init__(self, size=100, num_samples=30000, seed=None, training=True):  # todo:需要写一下另外类型的数据用于验证
        super(TBPPDataset, self).__init__()

        points = np.load('points50.npy').transpose((0, 2, 1))
        if training:
            self.dataset = torch.tensor(points[:, :, :])
            num_samples = points.shape[0]
        else:
            points = np.load('test50.npy').transpose((0, 2, 1))
            self.dataset = torch.tensor(points[:, :, :])
            num_samples = 90
        self.dynamic = torch.zeros(num_samples, 1, size)
        self.baseline = torch.tensor(np.load('solution50.npy'))
        self.num_nodes = size
        self.size = num_samples

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        return (self.dataset[idx], self.dynamic[idx], [], self.baseline[idx])


def update_mask(tour_idx, static, i):
    """Marks the visited city, so it can't be selected a second time."""
    batch_size, _, sequence_size = static.size()
    mask = torch.zeros(batch_size, sequence_size)
    for batch in range(batch_size):
        s_ = static[batch][0][i].item()
        c_ = static[batch][2][i].item()
        usage = [c_] * (i + 1)
        for item in range(i):
            bins = tour_idx[item][batch][0].item()
            if static[batch][0][item] <= s_ < static[batch][1][item]:
                usage[bins] += static[batch][2][item].item()
        for bins in range(i + 1):
            if usage[bins] <= 100:
                mask[batch][bins] = 1
        if mask[batch].sum() == 0:
            print('wrong')
    return mask


def update_mask_dynamic(dynamic, static, i):
    c = static[:, 2, i]
    c = c.unsqueeze(0).transpose(1, 0)
    mask = (dynamic >= c).int()
    mask[:, i + 1:] = 0
    return mask


def update_dynamic(tour_idx, static, i, mask=None, dynamic=None, tracker=None):
    batch_size, _, sequence_size = static.size()
    mask1 = torch.zeros(batch_size, sequence_size)
    dynamic = torch.zeros(batch_size, 1, sequence_size)
    for batch in range(batch_size):
        s_ = static[batch][0][i].item()
        e_ = static[batch][1][i].item()
        c_ = static[batch][2][i].item()
        # usage = [0] * (i + 1)
        # for item in range(i):
        #     bins = tour_idx[item][batch][0].item()
        #     if static[batch][0][item] <= s_ < static[batch][1][item]:
        #         usage[bins] += static[batch][2][item].item()
        if i>0:
            bin = tour_idx[i-1][batch][0].item()
            for item in range(i-1,sequence_size):
                if static[batch][0][item] < static[batch][1][i-1].item():
                    tracker[batch][bin][item] += static[batch][2][i-1].item()
                else:
                    break
        usage = tracker[batch,:,i]
        empty = np.argwhere(usage == 0)[0][0]
        mask1[batch][empty] = 1
        for bins in range(i + 1):
            remain = 100 - usage[bins]
            dynamic[batch][0][bins] = remain
            if c_ <= remain < 100:
                mask1[batch][bins] = 1
   #  dynamic = dynamic.unsqueeze(1)
    return dynamic, mask1


def reward(static, tour_indices):
    """
    Parameters
    ----------
    static: torch.FloatTensor containing static (e.g. x, y) data
    tour_indices: torch.IntTensor of size (batch_size, num_cities)

    Returns
    -------
    Euclidean distance between consecutive nodes on the route. of size
    (batch_size, num_cities)
    """

    # Convert the indices back into a tour
    batch_size, _, sequence_size = static.size()
    bins = torch.zeros(batch_size, sequence_size).scatter(1, tour_indices, 1)
    bins_num = bins.sum(1)
    return bins_num

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


class TiDataset(Dataset):

    def __init__(self, size=100, num_samples=30000, seed=None, training=True):
        super(TiDataset, self).__init__()
        if training:
            points = np.load('points50.npy').transpose((0, 2, 1))
            self.dataset = torch.cat((torch.zeros((points.shape[0], 3, 1)), torch.tensor(points[:, :, :])), 2)
            num_samples = points.shape[0]
        else:
            points = np.load('test50.npy').transpose((0, 2, 1))
            self.dataset = torch.cat((torch.zeros((points.shape[0], 3, 1)), torch.tensor(points[:, :3, :])), 2)
            num_samples = 90

        self.dynamic = torch.full((num_samples, 1, size), 100)
        self.dynamic = torch.cat((torch.zeros(num_samples, 1, 1), self.dynamic), 2) / 100
        self.baseline = torch.tensor(np.load('solution50.npy'))
        self.num_nodes = size
        self.size = num_samples

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        return self.dataset[idx], self.dynamic[idx], [], self.baseline[idx]


def update_mask(tour_idx, static, i, mask=None, dynamic=None):
    """Marks the visited city, so it can't be selected a second time."""
    batch_size, _, sequence_size = static.size()
    if not tour_idx:
        return mask
    if tour_idx[:, -1].eq(0).any():
        pass
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

    return mask


def update_dynamic(tour_idx, static, i, mask=None, dynamic=None, tracker=None):
    '''
    static: batch_size, 3, seq_len.  including start end c
    dynamic: batch, 1, seq_len. including remain for item start time
    tracker:seq_len. track the selection
    '''
    batch_size, _, sequence_size = static.size()
    if not tour_idx:
        tracker = np.zeros((batch_size, sequence_size))
        return dynamic, mask
    if tour_idx[-1].eq(0).all() and np.all(tracker == 1):
        return dynamic, torch.zeros(mask.size())
    dynamic1 = dynamic.clone()
    chosen_idx = tour_idx[-1]

    for batch in range(batch_size):
        ci = chosen_idx[batch]
        tracker[batch][ci] = 1
        if ci == 0:
            dynamic1[batch][0][:] = torch.ones(sequence_size)
            dynamic1[batch][0][0] = 0
            if np.all(tracker[batch][1:] == 1):  # all items get bins
                # mask[batch]=torch.zeros(sequence_size)
                # mask[batch][0] = 1
                pass
            else:

                mask[batch][:] = torch.zeros(sequence_size)
                mask[batch][np.where(tracker[batch] == 0)[0][0]] = 1
                # mask[batch][0] = 0
        else:
            c_ = static[batch][2][ci].item() / 100
            # dynamic1[batch][0][chosen_idx[batch]] = 0
            overlap = static[batch][0].lt(static[batch][1][ci]) & static[batch][0].ge(static[batch][0][ci])
            dynamic1[batch][0] = torch.where(overlap, dynamic[batch][0] - c_, dynamic[batch][0])
            # dynamic1[batch][1][np.arange(ci)] = 0 # todo:
            mask[batch][:] = torch.ones(sequence_size)
            mask[batch] = static[batch][2].le(dynamic1[batch][0] * 100).int()  # 在容量范围之内
            mask[batch][np.arange(ci + 1)] = 0  # 在当前物品之后
            mask[batch][tracker[batch] == 1] = 0  # 未被选择过
            # mask[batch][dynamic1[batch][0].eq(0)] = 0
            mask[batch][0] = 1  # 可以使用新箱子
    return dynamic1, mask


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
    tour_size = len(tour_indices[0])
    bins_num = []
    for i in range(batch_size):
        num = 0
        for j in range(tour_size - 1):
            if tour_indices[i][j + 1] == 0 and tour_indices[i][j] != 0:
                num += 1
        bins_num.append(num)
    return torch.tensor(bins_num).float()


def check(static, dynamic, tour_index):
    batch_size, tour_length = tour_index.size()
    for batch in range(batch_size):
        usage = [0] * 51
        k = np.zeros(51)
        for j in range(tour_length):
            if tour_index[batch][j] == 0:
                usage == [0] * 50
            else:
                b = tour_index[batch][j]
                if k[b] == 1:
                    print('visit twice', b)
                    return False
                k[b] = 1
                for x in range(b, 50):
                    if static[batch][0][x] < static[batch][1][b]:
                        usage[x] += static[batch][2][b]
                        if usage[x] > 100:
                            print('out capacity')
                            return False
        if (k[1:] == 0).any():
            print('miss')
            return False
    return True

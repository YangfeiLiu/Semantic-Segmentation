import numpy as np
import torch


def getWeight(weight_file):
    with open(weight_file, 'r') as f:
        line = f.readline()
        weight = line.split(' ')
        weight = [int(x) for x in weight]
        weight = np.array(weight)
        weight = weight / np.min(weight)
        weight = 1 / (np.log(weight) + 1.02)
        weight = torch.from_numpy(weight.astype(np.float32))
    return weight

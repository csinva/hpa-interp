import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import sys
from conv2dnp import conv2dnp
import copy
import cd


def gradient_times_input_scores(im, ind, model, device='cuda'):
    ind = torch.LongTensor([np.int(ind)]).to(device)
    if im.grad is not None:
        im.grad.data.zero_()
    pred = model(im)
    crit = nn.NLLLoss()
    loss = crit(pred, ind)
    loss.backward()
    res = im.grad * im
    return res.data.cpu().numpy()[0, 0]

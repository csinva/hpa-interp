import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import sys
import copy
import cv2
from cd import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def gradient_times_input_scores(im, ind, model):
    ind = torch.LongTensor([np.int(ind)]).to(device)
    if im.grad is not None:
        im.grad.data.zero_()
    pred = model(im)
    crit = nn.NLLLoss()
    loss = crit(pred, ind)
    loss.backward()
    res = im.grad * im
    return res.data.cpu().numpy()[0, 0]


# gradients global var
gradients = []
def save_gradient(grad):
    gradients.append(grad)


def grad_CAM(im, model, layer_ind=9, label_ind=2):
    im = im.to(device)
    if not im.requires_grad:
        im.requires_grad_(True)
    if im.grad is not None:
        im.grad.data.zero_()

    # get modules for network
    mods = forward_mods(model)
    mods.insert(0, norm) # add normalization in the beginning

    # get output of network, registering gradients from targetted intermediate layers
    feature_maps = []
    for i, mod in enumerate(mods[:12]):
        im = mod(im)
        if i == layer_ind:
            im.register_hook(save_gradient)
            feature_maps += [im]

    im0 = mods[12](im)
    im1 = mods[13](im)
    im = torch.cat((im0, im1), dim=1)
    im = im.view(im.size(0), -1)

    for i, mod in enumerate(mods[14:-1]):
        im = mod(im)
    im = im.view(im.size(0), -1)
    output = mods[18](im)

    # compute gradient w.r.t. feature maps
    one_hot = torch.zeros((1, output.size()[-1]), dtype = im.dtype)
    one_hot[0][label_ind] = 1
    one_hot = one_hot.requires_grad_(True)
    one_hot = torch.sum(one_hot.to(device) * output)
    one_hot.backward()

    # get gradient and feature map
    grads_val = gradients[-1].cpu().data.numpy()
    target = feature_maps[-1].cpu().data.numpy()[0,:]

    # get weight
    weights = np.mean(grads_val, axis = (2, 3))[0,:]

    # compute cam
    cam = torch.zeros(target.shape[1:], dtype = im.dtype).numpy()
    for i, w in enumerate(weights):
        cam += w * target[i,:,:]

    # relu
    # cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (1024, 1024))
    # cam = cam - np.min(cam)
    # cam = cam / np.max(cam)
    # cam = np.uint8(255 * cam)
    return cam, target

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from scipy.special import expit as sigmoid
from cd_propagate import *


def cd_densenet(blob, im_torch, model):
    '''CD Densenet
    '''
    # set up model
    model.eval()
    mods = forward_mods(model)

    # set up blobs
    device = torch.device("cuda:0")
    blob = blob.to(device)
    blob = torch.cuda.FloatTensor(blob)

    scores = []
    output = im_torch.clone().detach().to(device)
    # decompose
    relevant = blob * output
    irrelevant = (1 - blob) * output

    # propagate layers
    with torch.no_grad():
        # growth rate = 4; block config = (6, 12, 24, 16)
        # normalization step
        # mean = [0.074598, 0.050630, 0.050891, 0.076287]#rgby
        # std =  [0.122813, 0.085745, 0.129882, 0.119411]
        # for i in range(model.module.in_channels):
            # output[:,i,:,:] = (output[:,i,:,:] - mean[i]) / std[i]
        # decompose
        # relevant = blob * output
        # irrelevant = (1 - blob) * output
        relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, norm)
        scores.append((relevant.clone(), irrelevant.clone()))

        # initial convolution
        relevant, irrelevant = propagate_init_conv(relevant, irrelevant, mods[0])
        scores.append((relevant.clone(), irrelevant.clone()))

        if model.module.large:
            relevant, irrelevant = propagate_pooling(relevant, irrelevant, mods[1])
            scores.append((relevant.clone(), irrelevant.clone()))

        # denseblock1
        relevant, irrelevant = propagate_denseblock(relevant, irrelevant, mods[2])
        scores.append((relevant.clone(), irrelevant.clone()))

        # transition1
        relevant, irrelevant = propagate_transition(relevant, irrelevant, mods[3])
        scores.append((relevant.clone(), irrelevant.clone()))

        # denseblock2
        relevant, irrelevant = propagate_denseblock(relevant, irrelevant, mods[4])
        scores.append((relevant.clone(), irrelevant.clone()))

        # transition2
        relevant, irrelevant = propagate_transition(relevant, irrelevant, mods[5])
        scores.append((relevant.clone(), irrelevant.clone()))

        # denseblock3
        relevant, irrelevant = propagate_denseblock(relevant, irrelevant, mods[6])
        scores.append((relevant.clone(), irrelevant.clone()))

        # transition3
        relevant, irrelevant = propagate_transition(relevant, irrelevant, mods[7])
        scores.append((relevant.clone(), irrelevant.clone()))

        # denseblock4
        relevant, irrelevant = propagate_denseblock(relevant, irrelevant, mods[8])
        scores.append((relevant.clone(), irrelevant.clone()))

        # final batch norm
        relevant, irrelevant = propagate_batchnorm2d(relevant, irrelevant, mods[9])
        scores.append((relevant.clone(), irrelevant.clone()))

        # ReLU layer
        relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[10])
        scores.append((relevant.clone(), irrelevant.clone()))

        if model.module.dropout:
            # adaptive avgpooling and maxpooling
            rel0, irrel0 = propagate_avgpooling(relevant, irrelevant, mods[11])
            rel1, irrel1 = propagate_pooling(relevant, irrelevant, mods[12])
            relevant, irrelevant = torch.cat((rel0, rel1), dim=1), torch.cat((irrel0, irrel1), dim=1)
            scores.append((relevant.clone(), irrelevant.clone()))

            # reshape
            relevant = relevant.view(relevant.size(0), -1)
            irrelevant = irrelevant.view(irrelevant.size(0), -1)
            scores.append((relevant.clone(), irrelevant.clone()))

            # bn1 layer
            relevant, irrelevant = propagate_batchnorm2d(relevant, irrelevant, mods[13])
            scores.append((relevant.clone(), irrelevant.clone()))

            # fn1 layer
            relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[14])
            scores.append((relevant.clone(), irrelevant.clone()))

            # ReLU layer
            relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[15])
            scores.append((relevant.clone(), irrelevant.clone()))

            # bn2 layer
            relevant, irrelevant = propagate_batchnorm2d(relevant, irrelevant, mods[16])
            scores.append((relevant.clone(), irrelevant.clone()))

        # reshape
        relevant = relevant.view(relevant.size(0), -1)
        irrelevant = irrelevant.view(irrelevant.size(0), -1)

        # linear layer
        relevant, irrelevant = propagate_conv_linear(relevant, irrelevant, mods[17])
        scores.append((relevant.clone(), irrelevant.clone()))

    return relevant.cpu(), irrelevant.cpu(), scores


def forward_mods(model):
    # set up model
    model.eval()

    model_mods = list(model.modules())
    mods = []

    append_mod(model_mods[435], model.module.conv1, mods)
    if model.module.large:
        append_mod(False, model.module.maxpool, mods)
    append_mod(model_mods[436][0], model.module.encoder2[0], mods)
    append_mod(model_mods[437][0], model.module.encoder3[0], mods)
    append_mod(model_mods[437][1], model.module.encoder3[1], mods)
    append_mod(model_mods[438][0], model.module.encoder4[0], mods)
    append_mod(model_mods[438][1], model.module.encoder4[1], mods)
    append_mod(model_mods[439][0], model.module.encoder5[0], mods)
    append_mod(model_mods[439][1], model.module.encoder5[1], mods)
    append_mod(model_mods[439][2], model.module.encoder5[2], mods)
    append_mod(False, nn.ReLU(inplace=True), mods)
    if model.module.dropout:
        append_mod(False, nn.AdaptiveAvgPool2d(1), mods)
        append_mod(False, nn.AdaptiveMaxPool2d(1), mods)
        append_mod(model_mods[443], model.module.bn1, mods)
    #     append_mod(False, F.dropout(x, p=0.5, training=self.training), mods)
        append_mod(model_mods[444], model.module.fc1, mods)
        append_mod(False, nn.ReLU(inplace=True), mods)
        append_mod(model_mods[445], model.module.bn2, mods)
    #     append_mod(False, F.dropout(x, p=0.5, training=self.training), mods)
    else:
        append_mod(False, nn.AdaptiveAvgPool2d(1), mods)
    append_mod(model_mods[442], model.module.logit, mods)
    return mods


def append_mod(net_mod, mod, list_of_modules):
    if net_mod == False:
        list_of_modules.append(mod)
    else:
        Err = net_mod != mod
        if not Err:
            list_of_modules.append(net_mod)
        else:
            print('Error')
            raise ValueError


def forward_pass(im_torch, model):
    # set up model
    model.eval()
    mods = forward_mods(model)

    outputs = []
    device = torch.device("cuda:0")
    output = im_torch.clone().detach().to(device)

    mean = [0.074598, 0.050630, 0.050891, 0.076287]#rgby
    std =  [0.122813, 0.085745, 0.129882, 0.119411]
    for i in range(model.module.in_channels):
        output[:,i,:,:] = (output[:,i,:,:] - mean[i]) / std[i]
    outputs.append(output.clone())
    # propagate layers
    with torch.no_grad():
        for i, mod in enumerate(mods[:11]):
            output = mod(output)
            outputs.append(output.clone())

        output0 = mods[11](output)
        output1 = mods[12](output)
        output = torch.cat((output0, output1), dim=1)
        outputs.append(output.clone())

        output = output.view(output.size(0), -1)
        outputs.append(output.clone())

        for i, mod in enumerate(mods[13:-1]):
            output = mod(output)
            outputs.append(output.clone())

        output = output.view(output.size(0), -1)
        output = mods[17](output)
        outputs.append(output.clone())
    return output.cpu(), outputs


# normalize input tensors
def norm(inputs):
    x = inputs.clone().detach()
    mean = [0.074598, 0.050630, 0.050891, 0.076287]#rgby
    std =  [0.122813, 0.085745, 0.129882, 0.119411]
    for i in range(x.size()[1]):
        x[:,i,:,:] = (x[:,i,:,:] - mean[i]) / std[i]
    return x

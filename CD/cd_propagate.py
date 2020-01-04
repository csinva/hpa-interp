import torch
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from scipy.special import expit as sigmoid


# propagate linear layer
def propagate_linear(relevant, irrelevant, module):
    device = relevant.device
    dtype = relevant.dtype
    bias = module(torch.zeros(irrelevant.size(), dtype=dtype).to(device))
    rel = module(relevant) - bias
    irrel = module(irrelevant) - bias
    return rel + 0.0*bias, irrel + 1.0*bias
    # elementwise proportional
    prop_rel = torch.abs(rel)
    prop_irrel = torch.abs(irrel)
    prop_sum = prop_rel + prop_irrel
    prop_rel = torch.div(prop_rel, prop_sum)
    prop_irrel = torch.div(prop_irrel, prop_sum)
    return rel + torch.mul(prop_rel, bias), irrel + torch.mul(prop_irrel, bias)


# propagate conv layer
def propagate_conv(relevant, irrelevant, module):
    rel = module(relevant)
    irrel = module(irrelevant)
    return rel, irrel


# propagate ReLu nonlinearity
def propagate_relu(relevant, irrelevant, activation):
    '''propagate ReLu nonlinearity
    '''
    device = relevant.device
    dtype = relevant.dtype
    swap_inplace = False
    try:  # handles inplace
        if activation.inplace:
            swap_inplace = True
            activation.inplace = False
    except:
        pass
    # zeros = torch.zeros(relevant.size(), dtype=dtype).to(device)
    # rel_score = activation(relevant)
    # irrel_score = activation(relevant + irrelevant) - activation(relevant)
    indx = activation(relevant + irrelevant) == 0
    mask = torch.ones(relevant.size(), dtype=dtype).to(device)
    mask[indx] = 0
    rel_score = relevant * mask
    irrel_score = irrelevant * mask
    if swap_inplace:
        activation.inplace = True
    return rel_score, irrel_score


# propagate maxpooling operation
def propagate_pooling(relevant, irrelevant, pooler):
    # get both indices
    p = deepcopy(pooler)
    p.return_indices = True
    both, both_ind = p(relevant + irrelevant)

    # unpooling function
    def unpool(tensor, indices):
        batch_size, in_channels = indices.size()[:2]
        dtype = tensor.dtype
        output = torch.ones_like(indices, dtype=dtype)
        for i in range(batch_size):
            for j in range(in_channels):
                output[i,j] = tensor[i,j].view(-1)[indices[i,j].view(-1)].view(indices.size()[2:])
        return output

    rel, irrel = unpool(relevant, both_ind), unpool(irrelevant, both_ind)
    return rel, irrel


# propagate dropout operation
def propagate_dropout(relevant, irrelevant, module):
    return module(relevant), module(irrelevant)


# propagate avgpooling operation
def propagate_avgpooling(relevant, irrelevant, module):
    return module(relevant), module(irrelevant)


# propagate batchnorm2d operation
def propagate_batchnorm2d(relevant, irrelevant, module):
    # return propagate_linear(relevant, irrelevant, module)
    device = relevant.device
    dtype = relevant.dtype
    bias = module(torch.zeros(irrelevant.size(), dtype=dtype).to(device))
    rel = module(relevant) - bias
    irrel = module(irrelevant) - bias

    # elementwise proportional
    prop_rel = torch.abs(rel)
    prop_irrel = torch.abs(irrel)
    prop_sum = prop_rel + prop_irrel
    prop_rel = torch.div(prop_rel, prop_sum)
    prop_irrel = torch.div(prop_irrel, prop_sum)
    nan_ind = torch.isnan(prop_rel)
    prop_rel[nan_ind] = 0.5
    prop_irrel[nan_ind] = 0.5
    # return rel + torch.mul(prop_rel, bias), irrel + torch.mul(prop_irrel, bias)
    return rel + 0.0*bias, irrel + 1.0*bias


# propagate batchnorm1d operation
def propagate_batchnorm1d(relevant, irrelevant, module):
    return propagate_batchnorm2d(relevant, irrelevant, module)


# propagate initial convolution operation
def propagate_init_conv(relevant, irrelevant, init_modules):
    # (conv0): Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # (norm0): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (relu0): ReLU(inplace)
    # (pool0): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    relevant, irrelevant = propagate_conv(relevant, irrelevant, init_modules[0])
    relevant, irrelevant = propagate_batchnorm2d(relevant, irrelevant, init_modules[1])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, init_modules[2])
    relevant, irrelevant = propagate_pooling(relevant, irrelevant, init_modules[3])
    return relevant, irrelevant


# propagate denseblock
def propagate_denseblock(relevant, irrelevant, block_modules):
    num_layers = len(block_modules)
    for i in range(num_layers):
        relevant, irrelevant = propagate_denselayer(relevant, irrelevant, block_modules[i])
    return relevant, irrelevant


# propagate transition operation
def propagate_transition(relevant, irrelevant, trans_modules):
    # (norm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    # (relu): ReLU(inplace)
    # (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    # (pool): AvgPool2d(kernel_size=2, stride=2, padding=0)
    relevant, irrelevant = propagate_batchnorm2d(relevant, irrelevant, trans_modules[0])
    relevant, irrelevant = propagate_relu(relevant, irrelevant, trans_modules[1])
    relevant, irrelevant = propagate_conv(relevant, irrelevant, trans_modules[2])
    relevant, irrelevant = propagate_avgpooling(relevant, irrelevant, trans_modules[3])
    return relevant, irrelevant


# propagate denselayer
def propagate_denselayer(relevant, irrelevant, modules):
    rel, irrel = relevant.clone().detach(), irrelevant.clone().detach()
    # propagate dense layers
    rel, irrel = propagate_batchnorm2d(rel, irrel, modules[0])
    rel, irrel = propagate_relu(rel, irrel, modules[1])
    rel, irrel = propagate_conv(rel, irrel, modules[2])
    rel, irrel = propagate_batchnorm2d(rel, irrel, modules[3])
    rel, irrel = propagate_relu(rel, irrel, modules[4])
    rel, irrel = propagate_conv(rel, irrel, modules[5])
    return torch.cat([relevant, rel], 1), torch.cat([irrelevant, irrel], 1)


# def propagate_batchnorm2d(relevant, irrelevant, module):
#     dtype = relevant.dtype
#     bias = module(torch.zeros(irrelevant.size(), dtype=dtype).to(device))
#     rel = module(relevant) - bias
#     irrel = module(irrelevant) - bias
#
#     # elementwise proportional
#     prop_rel = torch.abs(rel)
#     prop_irrel = torch.abs(irrel)
#     prop_sum = prop_rel + prop_irrel
#     prop_rel = torch.div(prop_rel, prop_sum)
#     prop_irrel = torch.div(prop_irrel, prop_sum)
#     nan_ind = torch.isnan(prop_rel)
#     prop_rel[nan_ind] = 1.0
#     prop_irrel[nan_ind] = 0.0
#     return rel + torch.mul(prop_rel, bias), irrel + torch.mul(prop_irrel, bias)
#     # return rel + 0.0*bias, irrel + 1.0*bias

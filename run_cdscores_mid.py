import torch
import torch.nn as nn
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.autograd import Variable

import numpy as np
import pandas as pd
from tqdm import tqdm
from importlib import import_module
import sys, os
opj = os.path.join
ope = os.path.exists

this_dir = os.getcwd()
lib_paths = [opj(this_dir, 'bestfitting/protein_clean/src'), opj(this_dir, 'CD'), opj(this_dir, 'viz')]
for lib_path in lib_paths:
    if lib_path not in sys.path:
        sys.path.insert(0, lib_path)
import train_cls_net # import Protein class
from net import _init_paths
from config.config import * # set directory paths (DATA_DIR, RESULT_DIR etc)
from dataset import protein_dataset # import ProteinDataset class
from utils.augment_util import * # import augmentation functions
from net.loss_funcs.kaggle_metric import prob_to_result # import prob_to_result
from net.loss_funcs.kaggle_metric import get_probs_f1_score # import get_probs_f1_score

from cd_propagate import *
from cd import *
import viz
from copy import deepcopy
from matplotlib import gridspec
import pickle as pkl
import itertools
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import importlib
importlib.reload(viz)

module = 'densenet'
model_name = 'class_densenet121_large_dropout'
out_dir = 'external_crop1024_focal_slov_hardlog_clean'
train_batch_size = 36
test_batch_size = 12
scheduler_name = 'Adam45'
img_size = 1536
crop_size = 1024
in_channels = 4
gpus = '0' # which gpu to use

folds_num = 5
fold = 0
num_classes = 28

seed = 0
aug_version = 2
loss_name = 'FocalSymmetricLovaszHardLogLoss'
predict_aug = 'default,flipud,fliplr,transpose,flipud_lr,flipud_transpose,fliplr_transpose,flipud_lr_transpose' # augmentation functions

save_probs = True
clipnorm = True

# directory arguments
dir_args = {
    "split_dir": opj(DATA_DIR, "split"), # directory to locate labels
    "log_dir": opj(RESULT_DIR, "logs"),
    "subm_dir": opj(RESULT_DIR, "submissions"),
    "model_dir": opj(RESULT_DIR, "models"),
    "image_check_dir": opj(RESULT_DIR, "image_check"),
}

# data files
data_infos = {
    "model_level_name": "%s_i%d_aug%d_%dfolds/fold%d" % (model_name if out_dir is None else out_dir + '_' + model_name,
                                                             img_size, aug_version, folds_num, fold),
}

data_infos["train_split_file"] = "train_160.csv"
data_infos["valid_split_file"] = "valid_160.csv"
data_infos["test_split_file"] = "train_31072.csv"

# input of trainer.set_datasets
data_args = {
    "train_split_file": data_infos["train_split_file"],
    "valid_split_file": data_infos["valid_split_file"],
    "test_split_file": data_infos["test_split_file"],
    "model_level_name": data_infos["model_level_name"],
    "result_type": "test",
    'predict_aug':predict_aug,
}

# get Protein class
trainer = train_cls_net.Protein(dir_args,
                                train_batch_size=train_batch_size,
                                test_batch_size=test_batch_size,
                                seed=seed, img_size=img_size,in_channels=in_channels,
                                save_probs=save_probs,
                                aug_version=aug_version,
                                num_classes=num_classes,
                                crop_size=crop_size,
                                clipnorm=clipnorm)

# directory for densenet architecture
model = import_module("net.%s" % module)

# get densenet architecture pretrained on imagenet (model_name = class_densenet121_large_dropout)
net, scheduler, loss = model.get_model(model_name,
                                       num_classes,
                                       loss_name,
                                       scheduler_name=scheduler_name,
                                       in_channels=in_channels,
                                       )

# set directories for train, valid, test datasets and to save results
trainer.set_datasets(data_args)

# load model from model file
trainer.load_model(net=net, epoch=None)

# print model file
print('load model file from:', trainer.get_model_file())

# number of GPUs to use
n_gpu = trainer.setgpu(gpus)
net = trainer.set_data_parallel(net, n_gpu=n_gpu)

###########################
# which file to evaluate #
data_infos["test_split_file"] = "valid_160.csv"

data_args = {
    "train_split_file": data_infos["train_split_file"],
    "valid_split_file": data_infos["valid_split_file"],
    "test_split_file": data_infos["test_split_file"],
    "model_level_name": data_infos["model_level_name"],
    "result_type": "test",
    'predict_aug':predict_aug,
}

trainer.set_datasets(data_args)
###########################

# test dataset and dataloader
test_dataset = protein_dataset.ProteinDataset(trainer.test_split_file,
                                               img_size=trainer.img_size,
                                               is_trainset=True, # if True, save labels to self.labels
                                               return_label=True, # if True, return labels when indexing
                                               seed=trainer.seed,
                                               in_channels=trainer.in_channels,
                                               transform=None,
                                               crop_size=trainer.crop_size,
#                                                random_crop=trainer.seed!=0,
                                              random_crop=trainer.seed!=1,
                                               )

test_loader = protein_dataset.DataLoader(test_dataset,
                                         sampler=SequentialSampler(test_dataset),
                                         batch_size=trainer.test_batch_size,
#                                          batch_size=4,
                                         drop_last=False,
                                         num_workers=trainer.num_workers,
                                         pin_memory=True)

# list of augment functions
augments = trainer.predict_aug.split(',')

# pick one augmentation transform
augment_name = augments[0]

# set transform function
test_dataset.transform = [eval('augment_%s' % augment_name)]

# set directories for submission
epoch_name = 'epoch_final'
augment_name += '_seed%d'%seed
sub_dir = opj(trainer.subm_dir, epoch_name, augment_name)

# where to store the results
trainer.result_csv_file = opj(sub_dir, 'results_%s.csv.gz' % data_args['test_split_file'])
trainer.result_prob_fname = opj(sub_dir, 'prob_%s.npy' % data_args['test_split_file'])
trainer.extract_feat_fname = opj(sub_dir, 'extract_feats_%s.npz' % data_args['test_split_file'])

os.makedirs(sub_dir, exist_ok=True)

# use gpu
if trainer.gpu_flag:
    net.cuda()
# net eval mode
net.eval()

print('ready to evaluate')

def cd_activation_map(blob, feature_map, model):
    '''CD Densenet
    '''
    model.eval()

    scores = []
    device = torch.device("cuda:0")
    blob = blob.double().to(device)
    # blob = torch.cuda.FloatTensor(blob)

    output = feature_map.clone().detach().double().to(device)
    # decompose
    relevant = blob * output
    irrelevant = (1 - blob) * output

    # get modules for network
    mods = forward_mods(net)
    mods.insert(0, norm) # add normalization in the beginning

    # propagate layers
    with torch.no_grad():
        # growth rate = 4; block config = (6, 12, 24, 16)
        # final batch norm
        relevant, irrelevant = propagate_batchnorm2d(relevant, irrelevant, mods[10])
        scores.append((relevant.clone(), irrelevant.clone()))

        # ReLU layer
        relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[11])
        scores.append((relevant.clone(), irrelevant.clone()))

        if model.module.dropout:
            # adaptive avgpooling and maxpooling
            rel0, irrel0 = propagate_avgpooling(relevant, irrelevant, mods[12])
            rel1, irrel1 = propagate_pooling(relevant, irrelevant, mods[13])
            relevant, irrelevant = torch.cat((rel0, rel1), dim=1), torch.cat((irrel0, irrel1), dim=1)
            scores.append((relevant.clone(), irrelevant.clone()))

            # reshape
            relevant = relevant.view(relevant.size(0), -1)
            irrelevant = irrelevant.view(irrelevant.size(0), -1)
            scores.append((relevant.clone(), irrelevant.clone()))

            # bn1 layer
            relevant, irrelevant = propagate_batchnorm1d(relevant, irrelevant, mods[14])
            scores.append((relevant.clone(), irrelevant.clone()))

            # fn1 layer
            relevant, irrelevant = propagate_linear(relevant, irrelevant, mods[15])
            scores.append((relevant.clone(), irrelevant.clone()))

            # ReLU layer
            relevant, irrelevant = propagate_relu(relevant, irrelevant, mods[16])
            scores.append((relevant.clone(), irrelevant.clone()))

            # bn2 layer
            relevant, irrelevant = propagate_batchnorm1d(relevant, irrelevant, mods[17])
            scores.append((relevant.clone(), irrelevant.clone()))

        # reshape
        relevant = relevant.view(relevant.size(0), -1)
        irrelevant = irrelevant.view(irrelevant.size(0), -1)

        # linear layer
        relevant, irrelevant = propagate_linear(relevant, irrelevant, mods[18])
        scores.append((relevant.clone(), irrelevant.clone()))

    return relevant.cpu(), irrelevant.cpu(), scores

# true label
true_labels = pd.read_csv(trainer.test_split_file)['Target'].values

# get test image
for (images, labels, _) in tqdm(test_loader, total=int(np.ceil(test_dataset.num / test_batch_size))):
    pass

# test image
test_image = images[2:3]
test_label = labels[2:3]

# viz test image
# img_index = 0
# viz.viz_channels_separate(test_image, img_index)
# print(test_label)

# move to cuda
device = 'cuda'
im = test_image.clone().to(device)

# get modules for network
mods = forward_mods(net)
mods.insert(0, norm) # add normalization in the beginning

# get output of network, registering gradients from targetted intermediate layers
layer_ind = 9
feature_maps = []
with torch.no_grad():
    for i, mod in enumerate(mods[:12]):
        im = mod(im)
        if i == layer_ind:
            feature_maps += [im]

    im0 = mods[12](im)
    im1 = mods[13](im)
    im = torch.cat((im0, im1), dim=1)
    if layer_ind==13:
        feature_maps += [im]
    im = im.view(im.size(0), -1)

    for i, mod in enumerate(mods[14:-1]):
        im = mod(im)
    im = im.view(im.size(0), -1)
    output = mods[18](im)

# get feature map
target = feature_maps[-1]
print(target.size())

# convert to double type
net.double()
print('switch net to double')

# relevant scores
C, H, W = target.size()[1:]
rel_scores = torch.zeros(len(test_image), C, H, W, NUM_CLASSES)

with torch.no_grad():
    for c, h, w in itertools.product(range(C), range(H), range(W)):
        # set up blobs
        blob = torch.zeros(target.size())
        blob[:,c,h:(h+1),w:(w+1)] = 1

        relevant, irrelevant, _ = cd_activation_map(blob, target, net)
        if torch.norm(relevant + irrelevant - output.double().cpu()) > 1e-3:
            print('sum of cd scores do not match the original ouput at (channel,height,width) =', a(c,h,w))

        rel_scores[:,c,h:(h+1),w:(w+1),:] = relevant[:,None,None,:]

        print('\r iterations (channel,height,width) =', (c,h,w), end='')


test_image = test_image.cpu().numpy()
target = target.cpu().numpy()
rel_scores = rel_scores.cpu().numpy()
with open('test_image','wb') as file:
    pkl.dump(test_image, file)
with open('target','wb') as file:
    pkl.dump(target, file)
with open('rel_scores','wb') as file:
    pkl.dump(rel_scores, file)

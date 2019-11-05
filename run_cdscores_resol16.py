# import packages
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
lib_path = opj(this_dir, 'bestfitting/protein_clean/src')
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)
import train_cls_net # import Protein class
from net import _init_paths
from config.config import * # set directory paths (DATA_DIR, RESULT_DIR etc)
from dataset import protein_dataset # import ProteinDataset class
from utils.augment_util import * # import augmentation functions
from net.loss_funcs.kaggle_metric import prob_to_result # import prob_to_result
from net.loss_funcs.kaggle_metric import get_probs_f1_score # import get_probs_f1_score

### Set parameters and directories ###
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

### Load models and predict labels ###
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
                                               random_crop=trainer.seed!=0,
                                               )

test_loader = protein_dataset.DataLoader(test_dataset,
                                         sampler=SequentialSampler(test_dataset),
#                                          batch_size=trainer.test_batch_size,
                                         batch_size=4,
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

# count number of test points
n = 0

# get img_ids from dataset
img_ids = np.array(test_dataset.img_ids)

# list for probs
all_probs = []

with torch.no_grad():
#     for n_iter, (images, labels, indices) in tqdm(enumerate(test_loader, 0), total=int(np.ceil(test_dataset.num / trainer.test_batch_size))):
    for n_iter, (images, labels, indices) in tqdm(enumerate(test_loader, 0), total=int(np.ceil(test_dataset.num / 4))):
        batch_size = len(images)
        n += batch_size
        if trainer.gpu_flag:
            images = Variable(images.cuda(), volatile=True)
        else:
            images = Variable(images, volatile=True)

        outputs = net(images)
        logits = outputs

        probs = trainer.logits_to_probs(logits.data)
        all_probs += probs.cpu().numpy().reshape(-1).tolist() # collect all probs

# save probability vectors
all_probs = np.array(all_probs).reshape(-1, trainer.num_classes) # all_probs is an array of n-by-num_classes
if trainer.save_probs:
    np.save(trainer.result_prob_fname, all_probs)

# save predicted labels
df = prob_to_result(all_probs, img_ids) # prob_to_result located in net/loss_funcs/kaggle_metric.py; output pd.dataframe of img_ids and pred_list
df.to_csv(trainer.result_csv_file, index=False, compression='gzip')

# get F1-score
truth = pd.read_csv(trainer.test_split_file)
score = get_probs_f1_score(df, all_probs, truth, th=0.5)

print('macro f1 score:%.5f' % score)

pred_results = df.copy()
pred_results['Target'] = truth['Target'].values

### CD score ###
this_dir = os.getcwd()
lib_paths = [opj(this_dir, 'CD'), opj(this_dir, 'viz')]
for lib_path in lib_paths:
    if lib_path not in sys.path:
        sys.path.insert(0, lib_path)
from cd_propagate import *
from cd import *
# from viz import *
import viz
from copy import deepcopy
from matplotlib import gridspec
import pickle as pkl
import itertools

iterator = iter(test_loader)
# relevant scores1
n = 0

images = iterator.next()[0]

img_resize = 1024
superpixel_size = 16
h_num, w_num = int(img_resize/superpixel_size), int(img_resize/superpixel_size)

rel_scores = torch.zeros(4, in_channels, img_resize, img_resize, NUM_CLASSES)

with torch.no_grad():
    batch_size = len(images)
    n += batch_size
    for c, h, w in itertools.product(range(in_channels), range(h_num), range(w_num)):
        # set up blobs
        blob = torch.zeros(images.size())
        blob[:,c,h*superpixel_size:(h+1)*superpixel_size,w*superpixel_size:(w+1)*superpixel_size] = 1

        relevant, irrelevant, _ = cd_densenet(blob, images, net)
        output, _ = forward_pass(images, net)
        if torch.norm(relevant + irrelevant - output) > 1e-2:
            print('sum of cd scores do not match the original ouput at (channel,height,width) =', (c,h,w))

        rel_scores[:,c,h*superpixel_size:(h+1)*superpixel_size, \
                        w*superpixel_size:(w+1)*superpixel_size,:] = relevant[:,None,None,:]

        print('\r iterations (channel,height,width) =', (c,h,w), end='')

images = images.cpu().numpy()
rel_scores = rel_scores.cpu().numpy()
with open('images1.pkl','wb') as file:
    pkl.dump(images, file)
with open('cd_scores1.pkl','wb') as file:
    pkl.dump(rel_scores, file)

# relevant scores2
n = 0

images = iterator.next()[0]

img_resize = 1024
superpixel_size = 16
h_num, w_num = int(img_resize/superpixel_size), int(img_resize/superpixel_size)

rel_scores = torch.zeros(4, in_channels, img_resize, img_resize, NUM_CLASSES)

with torch.no_grad():
    batch_size = len(images)
    n += batch_size
    for c, h, w in itertools.product(range(in_channels), range(h_num), range(w_num)):
        # set up blobs
        blob = torch.zeros(images.size())
        blob[:,c,h*superpixel_size:(h+1)*superpixel_size,w*superpixel_size:(w+1)*superpixel_size] = 1

        relevant, irrelevant, _ = cd_densenet(blob, images, net)
        output, _ = forward_pass(images, net)
        if torch.norm(relevant + irrelevant - output) > 1e-2:
            print('sum of cd scores do not match the original ouput at (channel,height,width) =', (c,h,w))

        rel_scores[:,c,h*superpixel_size:(h+1)*superpixel_size, \
                        w*superpixel_size:(w+1)*superpixel_size,:] = relevant[:,None,None,:]

        print('\r iterations (channel,height,width) =', (c,h,w), end='')

images = images.cpu().numpy()
rel_scores = rel_scores.cpu().numpy()
with open('images2.pkl','wb') as file:
    pkl.dump(images, file)
with open('cd_scores2.pkl','wb') as file:
    pkl.dump(rel_scores, file)

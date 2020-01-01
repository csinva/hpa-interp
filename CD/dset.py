import torch
import numpy as np
import argparse
import sys, os
opj = os.path.join
ope = os.path.exists
this_dir = os.getcwd()
lib_path = opj(this_dir, '../bestfitting/protein_clean/src')
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)
from config.config import * # set directory paths (DATA_DIR, RESULT_DIR etc)
from importlib import import_module
from dataset import protein_dataset # import ProteinDataset class
from utils.augment_util import * # import augmentation functions


# Training settings
def get_args():
    parser = argparse.ArgumentParser(description='PyTorch HPA Example')
    parser.add_argument('--module', type=str, default='densenet', help='model (default: densenet)')
    parser.add_argument('--model_name', type=str, default='class_densenet121_large_dropout', help='model name (default: class_densenet121_large_dropout)')
    parser.add_argument('--out_dir', default='external_crop1024_focal_slov_hardlog_clean', type=str, help='output dir')
    parser.add_argument('--train_batch_size', default=36, type=int, help='train mini-batch size (default: 36)')
    parser.add_argument('--test_batch_size', default=12, type=int, help='test mini-batch size (default: 12)')
    parser.add_argument('--scheduler_name', default='Adam45', type=str, help='scheduler name (default: Adam45)')
    parser.add_argument('--img_size', default=1536, type=int, help='image size (default: 1536)')
    parser.add_argument('--crop_size', default=1024, type=int, help='crop size(default: 1024)')
    parser.add_argument('--in_channels', default=4, type=int, help='in channels (default: 4)')
    parser.add_argument('--gpus', default='0', type=str, help='use gpu (default: 0)') # which gpu to use
    parser.add_argument('--folds_num', default=5, type=int, help='number of folds (default: 5)')
    parser.add_argument('--fold', default=0, type=int, help='index of fold (default: 0)')
    parser.add_argument('--num_classes', default=28, type=int, help='number of classes (default: 28)')
    parser.add_argument('--seed', default=0, type=int, help='seed number (default: 0)')
    parser.add_argument('--aug_version', default=2, type=int, help='argument version(default: 2)')
    parser.add_argument('--loss_name', default='FocalSymmetricLovaszHardLogLoss', type=str, help='loss function FocalSymmetricLovaszHardLogLoss')
    parser.add_argument('--predict_aug', default='default,flipud,fliplr,transpose,flipud_lr,flipud_transpose,fliplr_transpose,flipud_lr_transpose',
                        type=str, help='')
    parser.add_argument('--save_probs', default=1, type=int, help='is save probability (default: 1)')
    parser.add_argument('--clipnorm', default=1, type=float, help='clip grad norm')
    return parser.parse_args("")


# setting dirs
def set_dirs(args, fname="train_31072.csv"):
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
        "model_level_name": "%s_i%d_aug%d_%dfolds/fold%d" % (args.model_name if args.out_dir is None else args.out_dir + '_' + args.model_name,
                                                                 args.img_size, args.aug_version, args.folds_num, args.fold),
    }

    data_infos["train_split_file"] = "train_160.csv"
    data_infos["valid_split_file"] = "valid_160.csv"
    data_infos["test_split_file"] = fname

    # input of trainer.set_datasets
    data_args = {
        "train_split_file": data_infos["train_split_file"],
        "valid_split_file": data_infos["valid_split_file"],
        "test_split_file": data_infos["test_split_file"],
        "model_level_name": data_infos["model_level_name"],
        "result_type": "test",
        'predict_aug':args.predict_aug,
    }
    return dir_args, data_args


# define dataset
def get_dataset(trainer):
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
    # list of augment functions
    augments = trainer.predict_aug.split(',')

    # pick one augmentation transform
    augment_name = augments[0]

    # set transform function
    test_dataset.transform = [eval('augment_%s' % augment_name)]

    return test_dataset


# load model
def load_densenet_pretrained(args, trainer, data_args):
    # directory for densenet architecture
    model = import_module("net.%s" % args.module)

    # get densenet architecture pretrained on imagenet (model_name = class_densenet121_large_dropout)
    net, scheduler, loss = model.get_model(args.model_name,
                                           args.num_classes,
                                           args.loss_name,
                                           scheduler_name=args.scheduler_name,
                                           in_channels=args.in_channels,
                                           )

    # set directories for train, valid, test datasets and to save results
    trainer.set_datasets(data_args)

    # load model from model file
    trainer.load_model(net=net, epoch=None)

    # print model file
    print('load model file from:', trainer.get_model_file())

    # number of GPUs to use
    n_gpu = trainer.setgpu(args.gpus)
    net = trainer.set_data_parallel(net, n_gpu=n_gpu)

    return net, scheduler, loss

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
from collections import Counter
import sys, os
opj = os.path.join

this_dir = os.getcwd()
lib_path = opj(this_dir, '../bestfitting/protein_clean/src')
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)
from config.config import *


def viz_channels_separate(imgs, img_index, rel_scores=None, pred_results=None):
    if pred_results is not None:
        print("Image ID=={}: true labels are [{}]; and predicted labels are [{}]".\
                        format(img_index, pred_results.loc[img_index, "Target"], pred_results.loc[img_index, "Predicted"]))
        print("True labels correspond to: ", end='')
        for location in pred_results.loc[img_index, "Target"].split():
            print('({}) '.format(LABEL_NAMES[int(location)]), end='')
        print("\nPredicted labels correspond to: ", end='')
        for location in pred_results.loc[img_index, "Predicted"].split():
            print('({})'.format(LABEL_NAMES[int(location)]), end='')

    if rel_scores is None:
        # get each image channel as a greyscale image
        img_red = imgs[img_index][0,:,:]
        img_green = imgs[img_index][1,:,:]
        img_blue = imgs[img_index][2,:,:]
        img_yellow = imgs[img_index][3,:,:]

        fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(12,7))
        ax[0].imshow(img_green, cmap="greens")
        ax[0].set_title("Protein of interest", fontsize=15)
        ax[1].imshow(img_red, cmap="reds")
        ax[1].set_title("Microtubules", fontsize=15)
        ax[2].imshow(img_blue, cmap="blues")
        ax[2].set_title("Nucleus", fontsize=15)
        ax[3].imshow(img_yellow, cmap="yellows")
        ax[3].set_title("ER", fontsize=15)
        for i in range(4):
            ax[i].set_xticklabels([])
            ax[i].set_yticklabels([])
            ax[i].tick_params(left=False, bottom=False)
        plt.subplots_adjust(wspace=0.02, hspace=0)
        plt.show()
    else:
        # get each image channel as a greyscale image
        img_red = imgs[img_index][0,:,:]
        img_green = imgs[img_index][1,:,:]
        img_blue = imgs[img_index][2,:,:]
        img_yellow = imgs[img_index][3,:,:]

        # get relative scores for each image channel
        rel_red = rel_scores[img_index][0,:,:]
        rel_green = rel_scores[img_index][1,:,:]
        rel_blue = rel_scores[img_index][2,:,:]
        rel_yellow = rel_scores[img_index][3,:,:]

        # get scales for reletive scores
        vmin = torch.min(rel_scores[img_index,:,:,:])
        vmax = torch.max(rel_scores[img_index,:,:,:])

        fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(12,6))
        ax[0, 0].imshow(img_green, cmap="greens")
        ax[0, 0].set_title("Protein of interest", fontsize=15)
        ax[0, 1].imshow(img_red, cmap="reds")
        ax[0, 1].set_title("Microtubules", fontsize=15)
        ax[0, 2].imshow(img_blue, cmap="blues")
        ax[0, 2].set_title("Nucleus", fontsize=15)
        ax[0, 3].imshow(img_yellow, cmap="yellows")
        ax[0, 3].set_title("ER", fontsize=15)
        im = ax[1, 0].imshow(rel_green, cmap="RdBu", vmin=vmin, vmax=vmax)
        im = ax[1, 1].imshow(rel_red, cmap="RdBu", vmin=vmin, vmax=vmax)
        im = ax[1, 2].imshow(rel_blue, cmap="RdBu", vmin=vmin, vmax=vmax)
        im = ax[1, 3].imshow(rel_yellow, cmap="RdBu", vmin=vmin, vmax=vmax)
        # im = ax[1, 0].imshow(rel_green, cmap="RdBu")
        # im = ax[1, 1].imshow(rel_red, cmap="RdBu")
        # im = ax[1, 2].imshow(rel_blue, cmap="RdBu")
        # im = ax[1, 3].imshow(rel_yellow, cmap="RdBu")
        for i in range(2):
            for j in range(4):
                ax[i, j].set_xticklabels([])
                ax[i, j].set_yticklabels([])
                ax[i, j].tick_params(left=False, bottom=False)
        plt.subplots_adjust(wspace=0, hspace=0)
        cbar_ax = fig.add_axes([0.92, 0.165, 0.015, 0.3])
        fig.colorbar(im, cax=cbar_ax)
        plt.show()


def viz_channels_combined(imgs, img_index, feature_importance=None):
    # get each image channel as a greyscale image
    img_red = imgs[img_index][0,:,:]
    img_green = imgs[img_index][1,:,:]
    img_blue = imgs[img_index][2,:,:]
    img_yellow = imgs[img_index][3,:,:]

    # create rgb images
    redRGB = cv2.merge((img_red.numpy(), np.zeros((1024, 1024),dtype='float32'), np.zeros((1024, 1024),dtype='float32')))
    greenRGB = cv2.merge((np.zeros((1024, 1024),dtype='float32'), img_green.numpy(), np.zeros((1024, 1024),dtype='float32')))
    blueRGB = cv2.merge((np.zeros((1024, 1024),dtype='float32'), np.zeros((1024, 1024),dtype='float32'), img_blue.numpy()))

    # add rgb images
    img = cv2.add(redRGB, greenRGB)
    img = cv2.add(img, blueRGB)

    #show result image
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7,7))
    img_viz = ax[0].imshow(img)
    cam = np.zeros(img.shape[:2]) if feature_importance is None else feature_importance
    cam_viz = ax[1].imshow(cam, cmap='RdBu')
    titles = ['red + green + blue', 'feature importance']
    for i, title in enumerate(titles):
        ax[i].set_title(title, fontsize=15)
        ax[i].set_xticklabels([])
        ax[i].set_yticklabels([])
        ax[i].tick_params(left=False, bottom=False)
    plt.subplots_adjust(wspace=0, hspace=0)
    cbar_ax = fig.add_axes([0.94, 0.325, 0.015, 0.3])
    fig.colorbar(cam_viz, cax=cbar_ax)
    plt.show()


#create custom color maps
cdict1 = {'red':   ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0)),

         'green': ((0.0,  0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'blue':  ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0))}

cdict2 = {'red':   ((0.0,  0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'green': ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0)),

         'blue':  ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0))}

cdict3 = {'red':   ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0)),

         'green': ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0)),

         'blue':  ((0.0,  0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0))}

cdict4 = {'red': ((0.0,  0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'green': ((0.0,  0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0,  1.0, 1.0)),

         'blue':  ((0.0,  0.0, 0.0),
                   (1.0,  0.0, 0.0))}

plt.register_cmap(name='greens', data=cdict1)
plt.register_cmap(name='reds', data=cdict2)
plt.register_cmap(name='blues', data=cdict3)
plt.register_cmap(name='yellows', data=cdict4)


#### gradCAM viz
# heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
# heatmap = np.float32(heatmap) / 255
# cam = heatmap + np.float32(img)
# cam = cam / np.max(cam)
# cam = np.uint8(255 * cam)


# from PIL import Image
# import matplotlib.pyplot as plt
# def load_image(path, id):
#     R = np.array(Image.open(path + id + '_red.png'))
#     G = np.array(Image.open(path + id + '_green.png'))
#     B = np.array(Image.open(path + id + '_blue.png'))
#     Y = np.array(Image.open(path + id + '_yellow.png'))
#     image = np.stack((R + Y/2, G + Y/2, B),-1).astype(np.float32)
#     return image
# my_image = load_image('/HPA/','0a1fe790-bac8-11e8-b2b7-ac1f6b6435d0')
# plt.imshow((my_image).astype(np.int))

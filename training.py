""" File for training loop, test loop, creating datasets and dataloaders and other helper functions """
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from sys import platform
from torch.utils.data import DataLoader
from Dev.LarsDev.csvdataset import collate
import torchvision.models as models
from Dev.LarsDev.csvdataset import myDatasetBase
import os
import random

g = torch.Generator()
g.manual_seed(0)
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def init_dataloaders(dataset_train,
                     dataset_test,
                     dataset_validation,
                     batchsize,
                     shuffle):
    """ Returns 3 dataloader for training, validation and testing the NN.
        Generator_seed set to 0 fo reproducibility."""
    g.manual_seed(0)
    train_dataloader = DataLoader(
        dataset_train,
        batch_size=batchsize,
        shuffle=shuffle,
        collate_fn=collate(),
        generator=g,
        drop_last=True)
    test_dataloader = DataLoader(
        dataset_test,
        batch_size=batchsize,
        shuffle=False,
        collate_fn=collate())
    validation_dataloader = DataLoader(
        dataset_validation,
        batch_size=batchsize,
        shuffle=False,
        collate_fn=collate())
    return train_dataloader, test_dataloader, validation_dataloader

def init_datasets(CSV_PATH, transforms_once, image_only_transforms):
    reset_randomness()
    dataset_train = myDatasetBase(
        csv_file=CSV_PATH,
        transforms_once=transforms_once,
        regenerate_aggregate_annotations=True,
        image_only_transforms=image_only_transforms,
        mode='train')
    dataset_test = myDatasetBase(
        csv_file=CSV_PATH,
        transforms_once=transforms_once,
        regenerate_aggregate_annotations=True,
        mode='test')
    dataset_validation = myDatasetBase(
        csv_file=CSV_PATH,
        transforms_once=transforms_once,
        regenerate_aggregate_annotations=True,
        mode='validation')
    return dataset_train, dataset_test, dataset_validation


def init_new_resnet(num_classes, device, pretrained=False):
    """ Returns a new resnet101 with always the same starting parameters.
    num_classes: # of classes/outputs
    """
    reset_randomness()
    net = models.resnet101(pretrained=pretrained, num_classes=num_classes).to(device)
    return net

def paths(csv_name):
    """ Returns paths of dataset list CSV and cache directory depending on
    operating system (windows or linux)
    Parameters:
        csv_name: name of the dataset list file to be used for training
    """
    if platform == 'linux':
        CSV_PATH = os.path.join('/home/lfeyerabend_host/workstation/Dev/LarsDev/dataset lists', csv_name)
        CACHE_PATH = '/data/face/recognition/lfeyerabend/cache'
    else:
        CSV_PATH = os.path.join(r'D:\Projects\prototyping\Dev\LarsDev\dataset lists', csv_name)
        CACHE_PATH = 'D:\Projects\FaceRecognition\Cache'
    return CSV_PATH, CACHE_PATH, platform

def show_input(inputs, n):
    """ Helper function to plot an input image while in the train_loop from the input[] tensor.
        """
    image = inputs[n].cpu()
    image = image/torch.max(image)
    image = image+1
    image = image/2
    fig = plt.figure(figsize=((max(n) * 5), 5))
    for i in n:
        sub = fig.add_subplot(1, (max(n)+1), i + 1)
        sub.imshow(image[i].permute(1, 2, 0), interpolation='nearest')

def show_source_img(data, n):
    """ Helper function to show the source img for a given image position in a batch (data) while in train_loop a batch is open
    Parameters:
        data: pass data-array from train-function
        n: image position in data-array to be displayed
    """
    img_path = data[2][n]['path']
    img = cv2.imread(img_path)
    cv2.imshow(f'{id}', img)

def reset_randomness():
    g = torch.Generator()
    g.manual_seed(0)
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

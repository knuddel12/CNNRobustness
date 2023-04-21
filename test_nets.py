import matplotlib
from sklearn.metrics import f1_score

matplotlib.use("agg")
import os
from copy import deepcopy
import torch
import numpy as np
from Utilities.Datasets.transforms import Load, ToTensor, cache_preprocessing
from training import init_dataloaders, init_new_resnet, paths, init_datasets, reset_randomness
from Dev.LarsDev.transforms import FaceAlignAndCrop, Grayscale
from kornia.augmentation import RandomGaussianBlur, RandomMotionBlur
from kornia import filters

""" This file is to test networks.
    Lists were created to test multiple networks at once. 
    The lists contain a label for a later plot and the file anem of
    a network to load it. 
    The test set conditions can be chosen in the marked section:
    Choose network/network list to be tested,
    """

reset_randomness()


def test_loop(MODEL_PATH,
              test_dataloader,
              sigma,
              net,
              blur_mode):
    """ This function takes the Model path / file name for the networks to be tested, test_dataloader for loading the
    test set,  """
    net.load_state_dict(torch.load(MODEL_PATH))
    net.eval()
    print("Last saved Model loaded.")
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    if blur_mode == 'median':
        blur_img = filters.MedianBlur([sigma, sigma])
    elif blur_mode == 'gaussian':
        blur_img = RandomGaussianBlur([9, 9], sigma=[sigma, sigma], p=1)
    elif blur_mode == 'box':
        blur_img = filters.BoxBlur(kernel_size=[sigma, sigma])
    elif blur_mode == 'motion':
        if sigma != 0:
            blur_img = RandomMotionBlur(kernel_size=int(sigma), angle=(360), direction=0, p=1)
    with torch.no_grad():
        for data in test_dataloader:
            imagesTest = torch.squeeze(data[0]).to(device)
            labelsTest = torch.squeeze(data[1]).to(device)
            if sigma != 0:
                imagesTest = blur_img(imagesTest)
            imagesTest = imagesTest - 0.5
            # calculate outputs by running images through the network
            outputsTest = net(imagesTest.float())
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputsTest.data, 1)
            total += labelsTest.size(0)
            correct += (predicted == labelsTest).sum().item()
            y_true.extend(labelsTest.tolist())
            y_pred.extend(predicted.tolist())
    accScore = np.around((100 * correct / total), decimals=1)
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f'Accuracy of the network on the test images: {accScore} %')
    print('sigma:', sigma)
    return accScore, f1


input_size = 112
batchsize = 100
num_classes = 530
params = {
    "lr": 0.005,
    "batchsize": batchsize,
    "input_sz": input_size * input_size * 3,
    "n_classes": num_classes}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
csv_name = 'facescrub_cleanup.csv'
CSV_PATH, CACHE_PATH, platform = paths(csv_name)
net = init_new_resnet(num_classes, device, pretrained=False)
transforms_once = [Load(cache_images=True, cache_folder=CACHE_PATH),
                   FaceAlignAndCrop(),
                   Grayscale(),
                   ToTensor(),
                   cache_preprocessing(CACHE_PATH)]
""" create dataset """
dataset_train, dataset_test, dataset_validation = init_datasets(CSV_PATH, transforms_once, image_only_transforms=None)
train_dataloader, test_dataloader, validation_dataloader = init_dataloaders(dataset_train,
                                                                            dataset_test,
                                                                            dataset_validation,
                                                                            batchsize,
                                                                            shuffle=True)

""" Lists filled with names of the networks which are to be tested.
    Facilitates testing multiple networks at once on a same test set. 
    Every entry consists of a name for the network for plotting later, 
    and the file name of the network itself. """

""" ##### Const blur ##### """

all_const_60epochs = [['motion_ks3_60', r'const_motion_blur3_lr0.005_epochs200_curr_60.pth'],
                      ['motion_ks5_60', r'const_motion_blur5_lr0.005_epochs200_curr_60.pth'],
                      ['motion_ks7_60', r'const_motion_blur7_lr0.005_epochs200_curr_60.pth'],
                      ['motion_ks9_60', r'const_motion_blur9_lr0.005_epochs200_curr_60.pth'],
                      ['median_ks3_60', r'const_median_blur3_lr0.005_epochs200_curr_60.pth'],
                      ['median_ks5_60', r'const_median_blur5_lr0.005_epochs200_curr_60.pth'],
                      ['median_ks7_60', r'const_median_blur7_lr0.005_epochs200_curr_60.pth'],
                      ['median_ks9_60', r'const_median_blur9_lr0.005_epochs200_curr_60.pth'],
                      ['box_ks3_60', r'const_box_blur3_lr0.005_epochs200_curr_60.pth'],
                      ['box_ks5_60', r'const_box_blur5_lr0.005_epochs200_curr_60.pth'],
                      ['box_ks7_60', r'const_box_blur7_lr0.005_epochs200_curr_60.pth'],
                      ['box_ks9_60', r'const_box_blur9_lr0.005_epochs200_curr_60.pth'],
                      ['gauss_ks3_60', r'const_blur0_lr0.005_epochs200_curr_60.pth'],
                      ['gauss_ks3_60', r'const_blur1_lr0.005_epochs200_curr_60.pth'],
                      ['gauss_ks3_60', r'const_blur2_lr0.005_epochs200_curr_60.pth'],
                      ['gauss_ks3_60', r'const_blur3_lr0.005_epochs200_curr_60.pth']]

const_gaussian_blur3_zehner = [['10', r'const_blur3_lr0.005_epochs200_curr_10.pth'],
                               ['20', r'const_blur3_lr0.005_epochs200_curr_20.pth'],
                               ['30', r'const_blur3_lr0.005_epochs200_curr_30.pth'],
                               ['40', r'const_blur3_lr0.005_epochs200_curr_40.pth'],
                               ['50', r'const_blur3_lr0.005_epochs200_curr_50.pth'],
                               ['60', r'const_blur3_lr0.005_epochs200_curr_60.pth'],
                               ['70', r'const_blur3_lr0.005_epochs200_curr_70.pth'],
                               ['80', r'const_blur3_lr0.005_epochs200_curr_80.pth'],
                               ['90', r'const_blur3_lr0.005_epochs200_curr_90.pth'],
                               ['100', r'const_blur3_lr0.005_epochs200_curr_100.pth'],
                               ['110', r'const_blur3_lr0.005_epochs200_curr_110.pth'],
                               ['120', r'const_blur3_lr0.005_epochs200_curr_120.pth'],
                               ['130', r'const_blur3_lr0.005_epochs200_curr_130.pth'],
                               ['140', r'const_blur3_lr0.005_epochs200_curr_140.pth'],
                               ['150', r'const_blur3_lr0.005_epochs200_curr_150.pth'],
                               ['160', r'const_blur3_lr0.005_epochs200_curr_160.pth'],
                               ['170', r'const_blur3_lr0.005_epochs200_curr_170.pth'],
                               ['180', r'const_blur3_lr0.005_epochs200_curr_180.pth'],
                               ['190', r'const_blur3_lr0.005_epochs200_curr_190.pth'],
                               ['200', r'const_blur3_lr0.005_epochs200_curr_200.pth']]
const_gaussian_blur2_zehner = [['10', r'const_blur2_lr0.005_epochs200_curr_10.pth'],
                               ['20', r'const_blur2_lr0.005_epochs200_curr_20.pth'],
                               ['30', r'const_blur2_lr0.005_epochs200_curr_30.pth'],
                               ['40', r'const_blur2_lr0.005_epochs200_curr_40.pth'],
                               ['50', r'const_blur2_lr0.005_epochs200_curr_50.pth'],
                               ['60', r'const_blur2_lr0.005_epochs200_curr_60.pth'],
                               ['70', r'const_blur2_lr0.005_epochs200_curr_70.pth'],
                               ['80', r'const_blur2_lr0.005_epochs200_curr_80.pth'],
                               ['90', r'const_blur2_lr0.005_epochs200_curr_90.pth'],
                               ['100', r'const_blur2_lr0.005_epochs200_curr_100.pth'],
                               ['110', r'const_blur2_lr0.005_epochs200_curr_110.pth'],
                               ['120', r'const_blur2_lr0.005_epochs200_curr_120.pth'],
                               ['130', r'const_blur2_lr0.005_epochs200_curr_130.pth'],
                               ['140', r'const_blur2_lr0.005_epochs200_curr_140.pth'],
                               ['150', r'const_blur2_lr0.005_epochs200_curr_150.pth'],
                               ['160', r'const_blur2_lr0.005_epochs200_curr_160.pth'],
                               ['170', r'const_blur2_lr0.005_epochs200_curr_170.pth'],
                               ['180', r'const_blur2_lr0.005_epochs200_curr_180.pth'],
                               ['190', r'const_blur2_lr0.005_epochs200_curr_190.pth'],
                               ['200', r'const_blur2_lr0.005_epochs200_curr_200.pth']]
const_gaussian_blur1_zehner = [['10', r'const_blur1_lr0.005_epochs200_curr_10.pth'],
                               ['20', r'const_blur1_lr0.005_epochs200_curr_20.pth'],
                               ['30', r'const_blur1_lr0.005_epochs200_curr_30.pth'],
                               ['40', r'const_blur1_lr0.005_epochs200_curr_40.pth'],
                               ['50', r'const_blur1_lr0.005_epochs200_curr_50.pth'],
                               ['60', r'const_blur1_lr0.005_epochs200_curr_60.pth'],
                               ['70', r'const_blur1_lr0.005_epochs200_curr_70.pth'],
                               ['80', r'const_blur1_lr0.005_epochs200_curr_80.pth'],
                               ['90', r'const_blur1_lr0.005_epochs200_curr_90.pth'],
                               ['100', r'const_blur1_lr0.005_epochs200_curr_100.pth'],
                               ['110', r'const_blur1_lr0.005_epochs200_curr_110.pth'],
                               ['120', r'const_blur1_lr0.005_epochs200_curr_120.pth'],
                               ['130', r'const_blur1_lr0.005_epochs200_curr_130.pth'],
                               ['140', r'const_blur1_lr0.005_epochs200_curr_140.pth'],
                               ['150', r'const_blur1_lr0.005_epochs200_curr_150.pth'],
                               ['160', r'const_blur1_lr0.005_epochs200_curr_160.pth'],
                               ['170', r'const_blur1_lr0.005_epochs200_curr_170.pth'],
                               ['180', r'const_blur1_lr0.005_epochs200_curr_180.pth'],
                               ['190', r'const_blur1_lr0.005_epochs200_curr_190.pth'],
                               ['200', r'const_blur1_lr0.005_epochs200_curr_200.pth']]
const_gaussian_blur0_zehner = [['10', r'const_blur0_lr0.005_epochs200_curr_10.pth'],
                               ['20', r'const_blur0_lr0.005_epochs200_curr_20.pth'],
                               ['30', r'const_blur0_lr0.005_epochs200_curr_30.pth'],
                               ['40', r'const_blur0_lr0.005_epochs200_curr_40.pth'],
                               ['50', r'const_blur0_lr0.005_epochs200_curr_50.pth'],
                               ['60', r'const_blur0_lr0.005_epochs200_curr_60.pth'],
                               ['70', r'const_blur0_lr0.005_epochs200_curr_70.pth'],
                               ['80', r'const_blur0_lr0.005_epochs200_curr_80.pth'],
                               ['90', r'const_blur0_lr0.005_epochs200_curr_90.pth'],
                               ['100', r'const_blur0_lr0.005_epochs200_curr_100.pth'],
                               ['110', r'const_blur0_lr0.005_epochs200_curr_110.pth'],
                               ['120', r'const_blur0_lr0.005_epochs200_curr_120.pth'],
                               ['130', r'const_blur0_lr0.005_epochs200_curr_130.pth'],
                               ['140', r'const_blur0_lr0.005_epochs200_curr_140.pth'],
                               ['150', r'const_blur0_lr0.005_epochs200_curr_150.pth'],
                               ['160', r'const_blur0_lr0.005_epochs200_curr_160.pth'],
                               ['170', r'const_blur0_lr0.005_epochs200_curr_170.pth'],
                               ['180', r'const_blur0_lr0.005_epochs200_curr_180.pth'],
                               ['190', r'const_blur0_lr0.005_epochs200_curr_190.pth'],
                               ['200', r'const_blur0_lr0.005_epochs200_curr_200.pth']]

const_motion_blur9_zehner = [['10', r'const_motion_blur9_lr0.005_epochs200_curr_10.pth'],
                             ['20', r'const_motion_blur9_lr0.005_epochs200_curr_20.pth'],
                             ['30', r'const_motion_blur9_lr0.005_epochs200_curr_30.pth'],
                             ['40', r'const_motion_blur9_lr0.005_epochs200_curr_40.pth'],
                             ['50', r'const_motion_blur9_lr0.005_epochs200_curr_50.pth'],
                             ['60', r'const_motion_blur9_lr0.005_epochs200_curr_60.pth'],
                             ['70', r'const_motion_blur9_lr0.005_epochs200_curr_70.pth'],
                             ['80', r'const_motion_blur9_lr0.005_epochs200_curr_80.pth'],
                             ['90', r'const_motion_blur9_lr0.005_epochs200_curr_90.pth'],
                             ['100', r'const_motion_blur9_lr0.005_epochs200_curr_100.pth'],
                             ['110', r'const_motion_blur9_lr0.005_epochs200_curr_110.pth'],
                             ['120', r'const_motion_blur9_lr0.005_epochs200_curr_120.pth'],
                             ['130', r'const_motion_blur9_lr0.005_epochs200_curr_130.pth'],
                             ['140', r'const_motion_blur9_lr0.005_epochs200_curr_140.pth'],
                             ['150', r'const_motion_blur9_lr0.005_epochs200_curr_150.pth'],
                             ['160', r'const_motion_blur9_lr0.005_epochs200_curr_160.pth'],
                             ['170', r'const_motion_blur9_lr0.005_epochs200_curr_170.pth'],
                             ['180', r'const_motion_blur9_lr0.005_epochs200_curr_180.pth'],
                             ['190', r'const_motion_blur9_lr0.005_epochs200_curr_190.pth'],
                             ['200', r'const_motion_blur9_lr0.005_epochs200_curr_200.pth']]
const_motion_blur7_zehner = [['10', r'const_motion_blur7_lr0.005_epochs200_curr_10.pth'],
                             ['20', r'const_motion_blur7_lr0.005_epochs200_curr_20.pth'],
                             ['30', r'const_motion_blur7_lr0.005_epochs200_curr_30.pth'],
                             ['40', r'const_motion_blur7_lr0.005_epochs200_curr_40.pth'],
                             ['50', r'const_motion_blur7_lr0.005_epochs200_curr_50.pth'],
                             ['60', r'const_motion_blur7_lr0.005_epochs200_curr_60.pth'],
                             ['70', r'const_motion_blur7_lr0.005_epochs200_curr_70.pth'],
                             ['80', r'const_motion_blur7_lr0.005_epochs200_curr_80.pth'],
                             ['90', r'const_motion_blur7_lr0.005_epochs200_curr_90.pth'],
                             ['100', r'const_motion_blur7_lr0.005_epochs200_curr_100.pth'],
                             ['110', r'const_motion_blur7_lr0.005_epochs200_curr_110.pth'],
                             ['120', r'const_motion_blur7_lr0.005_epochs200_curr_120.pth'],
                             ['130', r'const_motion_blur7_lr0.005_epochs200_curr_130.pth'],
                             ['140', r'const_motion_blur7_lr0.005_epochs200_curr_140.pth'],
                             ['150', r'const_motion_blur7_lr0.005_epochs200_curr_150.pth'],
                             ['160', r'const_motion_blur7_lr0.005_epochs200_curr_160.pth'],
                             ['170', r'const_motion_blur7_lr0.005_epochs200_curr_170.pth'],
                             ['180', r'const_motion_blur7_lr0.005_epochs200_curr_180.pth'],
                             ['190', r'const_motion_blur7_lr0.005_epochs200_curr_190.pth'],
                             ['200', r'const_motion_blur7_lr0.005_epochs200_curr_200.pth']]
const_motion_blur5_zehner = [['10', r'const_motion_blur5_lr0.005_epochs200_curr_10.pth'],
                             ['20', r'const_motion_blur5_lr0.005_epochs200_curr_20.pth'],
                             ['30', r'const_motion_blur5_lr0.005_epochs200_curr_30.pth'],
                             ['40', r'const_motion_blur5_lr0.005_epochs200_curr_40.pth'],
                             ['50', r'const_motion_blur5_lr0.005_epochs200_curr_50.pth'],
                             ['60', r'const_motion_blur5_lr0.005_epochs200_curr_60.pth'],
                             ['70', r'const_motion_blur5_lr0.005_epochs200_curr_70.pth'],
                             ['80', r'const_motion_blur5_lr0.005_epochs200_curr_80.pth'],
                             ['90', r'const_motion_blur5_lr0.005_epochs200_curr_90.pth'],
                             ['100', r'const_motion_blur5_lr0.005_epochs200_curr_100.pth'],
                             ['110', r'const_motion_blur5_lr0.005_epochs200_curr_110.pth'],
                             ['120', r'const_motion_blur5_lr0.005_epochs200_curr_120.pth'],
                             ['130', r'const_motion_blur5_lr0.005_epochs200_curr_130.pth'],
                             ['140', r'const_motion_blur5_lr0.005_epochs200_curr_140.pth'],
                             ['150', r'const_motion_blur5_lr0.005_epochs200_curr_150.pth'],
                             ['160', r'const_motion_blur5_lr0.005_epochs200_curr_160.pth'],
                             ['170', r'const_motion_blur5_lr0.005_epochs200_curr_170.pth'],
                             ['180', r'const_motion_blur5_lr0.005_epochs200_curr_180.pth'],
                             ['190', r'const_motion_blur5_lr0.005_epochs200_curr_190.pth'],
                             ['200', r'const_motion_blur5_lr0.005_epochs200_curr_200.pth']]
const_motion_blur3_zehner = [['10', r'const_motion_blur3_lr0.005_epochs200_curr_10.pth'],
                             ['20', r'const_motion_blur3_lr0.005_epochs200_curr_20.pth'],
                             ['30', r'const_motion_blur3_lr0.005_epochs200_curr_30.pth'],
                             ['40', r'const_motion_blur3_lr0.005_epochs200_curr_40.pth'],
                             ['50', r'const_motion_blur3_lr0.005_epochs200_curr_50.pth'],
                             ['60', r'const_motion_blur3_lr0.005_epochs200_curr_60.pth'],
                             ['70', r'const_motion_blur3_lr0.005_epochs200_curr_70.pth'],
                             ['80', r'const_motion_blur3_lr0.005_epochs200_curr_80.pth'],
                             ['90', r'const_motion_blur3_lr0.005_epochs200_curr_90.pth'],
                             ['100', r'const_motion_blur3_lr0.005_epochs200_curr_100.pth'],
                             ['110', r'const_motion_blur3_lr0.005_epochs200_curr_110.pth'],
                             ['120', r'const_motion_blur3_lr0.005_epochs200_curr_120.pth'],
                             ['130', r'const_motion_blur3_lr0.005_epochs200_curr_130.pth'],
                             ['140', r'const_motion_blur3_lr0.005_epochs200_curr_140.pth'],
                             ['150', r'const_motion_blur3_lr0.005_epochs200_curr_150.pth'],
                             ['160', r'const_motion_blur3_lr0.005_epochs200_curr_160.pth'],
                             ['170', r'const_motion_blur3_lr0.005_epochs200_curr_170.pth'],
                             ['180', r'const_motion_blur3_lr0.005_epochs200_curr_180.pth'],
                             ['190', r'const_motion_blur3_lr0.005_epochs200_curr_190.pth'],
                             ['200', r'const_motion_blur3_lr0.005_epochs200_curr_200.pth']]

""" ##### Step-wise Blur Change ##### """

# Two-split approaches: Gaussian high to low, low to high, high only, low only (baseline)
two_split_gaussian = [['baseline', r'baseline.pth'],
                      ['high_to_low', r'2split_blur_3_to_0_lr0.005_epochs400.pth'],
                      ['low_to_high', r'2split_blur_0_to_3_lr0.005_epochs400.pth'],
                      ['high_only', r'2split_blur_3_to_3_lr0.005_epochs400.pth']]

continued_3_to_0 = [['10', 'const_blur0_lr0.005_epochs200_curr_10.pth'],
                    ['20', 'const_blur0_lr0.005_epochs200_curr_20.pth'],
                    ['30', 'const_blur0_lr0.005_epochs200_curr_30.pth'],
                    ['40', 'const_blur0_lr0.005_epochs200_curr_40.pth'],
                    ['50', 'const_blur0_lr0.005_epochs200_curr_50.pth'],
                    ['60', 'const_blur0_lr0.005_epochs200_curr_60.pth'],
                    ['70', 'const_blur0_lr0.005_epochs200_curr_70.pth'],
                    ['80', 'const_blur0_lr0.005_epochs200_curr_80.pth'],
                    ['90', 'const_blur0_lr0.005_epochs200_curr_90.pth'],
                    ['100', 'const_blur0_lr0.005_epochs200_curr_100.pth'],
                    ['110', 'const_blur0_lr0.005_epochs200_curr_110.pth'],
                    ['120', 'const_blur0_lr0.005_epochs200_curr_120.pth'],
                    ['200', 'const_blur0_lr0.005_epochs200_curr_200.pth']]
continued_3_to_15 = [['10', 'const_blur1.5_lr0.005_epochs200_curr_10.pth'],
                     ['20', 'const_blur1.5_lr0.005_epochs200_curr_20.pth'],
                     ['30', 'const_blur1.5_lr0.005_epochs200_curr_30.pth'],
                     ['40', 'const_blur1.5_lr0.005_epochs200_curr_40.pth'],
                     ['50', 'const_blur1.5_lr0.005_epochs200_curr_50.pth'],
                     ['60', 'const_blur1.5_lr0.005_epochs200_curr_60.pth'],
                     ['70', 'const_blur1.5_lr0.005_epochs200_curr_70.pth'],
                     ['80', 'const_blur1.5_lr0.005_epochs200_curr_80.pth'],
                     ['90', 'const_blur1.5_lr0.005_epochs200_curr_90.pth'],
                     ['100', 'const_blur1.5_lr0.005_epochs200_curr_100.pth'],
                     ['110', 'const_blur1.5_lr0.005_epochs200_curr_110.pth'],
                     ['120', 'const_blur1.5_lr0.005_epochs200_curr_120.pth'],
                     ['200', 'const_blur1.5_lr0.005_epochs200_curr_200.pth']]

all_high_to_low = [['baseline', r'AdamW_zerocentered_baseline_resnet101_facescrub_0.005lr_400epochs_eps_1e-6.pth'],
                   ['high_to_low_motion', r'2split_motion_blur_9_to_0_angle360_lr0.005_epochs400.pth'],
                   ['high_to_low_gaussian', r'2split_blur_3_to_0_lr0.005_epochs400.pth'],
                   ['high_to_low_median', r'2split_median_blur_7_to_0_lr0.005_epochs400.pth'],
                   ['high_to_low_box', r'2split_box_blur_7_to_0_lr0.005_epochs400.pth']]

# Three-split approaches: Gaussian high to low, low to high, high only, low only (baseline)
continued_probability_3_to_15_to_0 = [['10', 'const_blur3_to_sigma1.5_to_sigma0_p0.66_lr0.005_epochs200_curr_10.pth'],
                                      ['20', 'const_blur3_to_sigma1.5_to_sigma0_p0.66_lr0.005_epochs200_curr_20.pth'],
                                      ['30', 'const_blur3_to_sigma1.5_to_sigma0_p0.66_lr0.005_epochs200_curr_30.pth'],
                                      ['40', 'const_blur3_to_sigma1.5_to_sigma0_p0.66_lr0.005_epochs200_curr_40.pth'],
                                      ['50', 'const_blur3_to_sigma1.5_to_sigma0_p0.66_lr0.005_epochs200_curr_50.pth'],
                                      ['60', 'const_blur3_to_sigma1.5_to_sigma0_p0.66_lr0.005_epochs200_curr_60.pth'],
                                      ['70', 'const_blur3_to_sigma1.5_to_sigma0_p0.66_lr0.005_epochs200_curr_70.pth'],
                                      ['80', 'const_blur3_to_sigma1.5_to_sigma0_p0.66_lr0.005_epochs200_curr_80.pth'],
                                      ['90', 'const_blur3_to_sigma1.5_to_sigma0_p0.66_lr0.005_epochs200_curr_90.pth'],
                                      ['100', 'const_blur3_to_sigma1.5_to_sigma0_p0.66_lr0.005_epochs200_curr_100.pth'],
                                      ['110', 'const_blur3_to_sigma1.5_to_sigma0_p0.66_lr0.005_epochs200_curr_110.pth'],
                                      ['120', 'const_blur3_to_sigma1.5_to_sigma0_p0.66_lr0.005_epochs200_curr_120.pth'],
                                      ['200', 'const_blur3_to_sigma1.5_to_sigma0_p0.66_lr0.005_epochs200_curr_200.pth']]
continued_const_motion_9_to_5_to_0 = [
    ['10', 'const_motion_blur_training_continued_ks9_to_5_to_0_200_epochs_curr_10.pth'],
    ['20', 'const_motion_blur_training_continued_ks9_to_5_to_0_200_epochs_curr_20.pth'],
    ['30', 'const_motion_blur_training_continued_ks9_to_5_to_0_200_epochs_curr_30.pth'],
    ['40', 'const_motion_blur_training_continued_ks9_to_5_to_0_200_epochs_curr_40.pth'],
    ['50', 'const_motion_blur_training_continued_ks9_to_5_to_0_200_epochs_curr_50.pth'],
    ['60', 'const_motion_blur_training_continued_ks9_to_5_to_0_200_epochs_curr_60.pth'],
    ['70', 'const_motion_blur_training_continued_ks9_to_5_to_0_200_epochs_curr_70.pth'],
    ['80', 'const_motion_blur_training_continued_ks9_to_5_to_0_200_epochs_curr_80.pth'],
    ['200', 'const_motion_blur_training_continued_ks9_to_5_to_0_200_epochs_curr_200.pth']]

""" ##### Epoch-wise Blur Change ##### """

continued_iterative = [['10', 'iterative_blur_3_lr0.005_epochs400_new.pth_curr_70_iterative.pth'],
                       ['20', 'iterative_blur_3_lr0.005_epochs400_new.pth_curr_80_iterative.pth'],
                       ['40', 'iterative_blur_3_lr0.005_epochs400_new.pth_curr_100_iterative.pth'],
                       ['60', 'iterative_blur_3_lr0.005_epochs400_new.pth_curr_120_iterative.pth'],
                       ['80', 'iterative_blur_3_lr0.005_epochs400_new.pth_curr_140_iterative.pth'],
                       ['100', 'iterative_blur_3_lr0.005_epochs400_new.pth_curr_160_iterative.pth'],
                       ['120', 'iterative_blur_3_lr0.005_epochs400_new.pth_curr_180_iterative.pth'],
                       ['140', 'iterative_blur_3_lr0.005_epochs400_new.pth_curr_200_iterative.pth'],
                       ['160', 'iterative_blur_3_lr0.005_epochs400_new.pth_curr_220_iterative.pth'],
                       ['180', 'iterative_blur_3_lr0.005_epochs400_new.pth_curr_240_iterative.pth'],
                       ['200', 'iterative_blur_3_lr0.005_epochs400_new.pth_curr_260_iterative.pth'],
                       ['340', 'iterative_blur_3_lr0.005_epochs400_new.pth_curr_400_iterative.pth']]

""" ##### Random Blur ##### """

random_blur_gaussian = [['random', r'random_blur3_lr0.005_epochs400.pth']]
continued_probability_3_to_0 = [['10', 'const_blur3_to_sigma0_p0.5_lr0.005_epochs200_curr_10.pth'],
                                ['20', 'const_blur3_to_sigma0_p0.5_lr0.005_epochs200_curr_20.pth'],
                                ['30', 'const_blur3_to_sigma0_p0.5_lr0.005_epochs200_curr_30.pth'],
                                ['40', 'const_blur3_to_sigma0_p0.5_lr0.005_epochs200_curr_40.pth'],
                                ['50', 'const_blur3_to_sigma0_p0.5_lr0.005_epochs200_curr_50.pth'],
                                ['60', 'const_blur3_to_sigma0_p0.5_lr0.005_epochs200_curr_60.pth'],
                                ['70', 'const_blur3_to_sigma0_p0.5_lr0.005_epochs200_curr_70.pth'],
                                ['80', 'const_blur3_to_sigma0_p0.5_lr0.005_epochs200_curr_80.pth'],
                                ['90', 'const_blur3_to_sigma0_p0.5_lr0.005_epochs200_curr_90.pth'],
                                ['100', 'const_blur3_to_sigma0_p0.5_lr0.005_epochs200_curr_100.pth'],
                                ['110', 'const_blur3_to_sigma0_p0.5_lr0.005_epochs200_curr_110.pth'],
                                ['120', 'const_blur3_to_sigma0_p0.5_lr0.005_epochs200_curr_120.pth'],
                                ['200', 'const_blur3_to_sigma0_p0.5_lr0.005_epochs200_curr_200.pth']]
continued_motion_probability_9_to_0 = [['10', 'const_motion_blur_training_continued_ks9_p05_200_epochs_curr_10.pth'],
                                       ['20', 'const_motion_blur_training_continued_ks9_p05_200_epochs_curr_20.pth'],
                                       ['40', 'const_motion_blur_training_continued_ks9_p05_200_epochs_curr_40.pth'],
                                       ['60', 'const_motion_blur_training_continued_ks9_p05_200_epochs_curr_60.pth'],
                                       ['80', 'const_motion_blur_training_continued_ks9_p05_200_epochs_curr_80.pth'],
                                       ['100', 'const_motion_blur_training_continued_ks9_p05_200_epochs_curr_100.pth'],
                                       ['200', 'const_motion_blur_training_continued_ks9_p05_200_epochs_curr_200.pth']]



if platform == 'linux':
    abs_path = r'/home/lfeyerabend_host/workstation/Dev/LarsDev/'
else:
    abs_path = r'D:\Projects\prototyping\Dev\LarsDev'
# Defining range of blur degree of test images
gaussian_params = np.arange(0, 5.5, 0.5)
median_params = np.array([0, 3, 5, 7, 9, 11, 13])
box_params = np.array([0, 3, 5, 7, 9, 11, 13])
motion_params = np.array([0, 3, 5, 7, 9, 11, 13])

#######################
""" Edit Test here """
test_list = [
    'only_baseline']  # Choose testlist with networks to be tested; alternatively, in the line below a single network
blur_mode = 'motion'  # Choose test image blur (gaussian, box, median, motion)
directory = ''  # directory where the nets are saved

""" Until here"""
#######################

if blur_mode == 'gaussian':
    blur_params = gaussian_params
elif blur_mode == 'median':
    blur_params = median_params
elif blur_mode == 'box':
    blur_params = box_params
elif blur_mode == 'motion':
    blur_params = motion_params

""" Test accuracies. Loop through the test list of networks """
for network_type in test_list:
    testlist = eval(network_type)
    for i, name in enumerate(testlist):
        testlist[i][1] = os.path.join(abs_path, directory, name[1])
    print(f"Testing {network_type} Nets on {blur_mode} blur.")
    f1scores = deepcopy(testlist)
    for i in range(len(testlist)):
        for sigma in blur_params:
            reset_randomness()
            acc, f1score = test_loop(MODEL_PATH=testlist[i][1],
                                     test_dataloader=test_dataloader,
                                     sigma=sigma,
                                     net=net,
                                     blur_mode=blur_mode)
            testlist[i].append(acc)
            f1scores[i].append(f1score)
    testlist.insert(0, ['name', 'file'])
    f1scores.insert(0, ['name', 'file'])
    for k in blur_params:
        testlist[0].append(k)
        f1scores[0].append(k)
    # Save test results as .txt file
    np.savetxt(os.path.join(abs_path, 'testresults/', f'acc_results_{network_type}_tested_on_{blur_mode}_blur.txt'),
               np.asarray(testlist), fmt="%s")
    np.savetxt(os.path.join(abs_path, 'testresults/', f'f1_results_{network_type}_tested_on_{blur_mode}_blur.txt'),
               np.asarray(testlist), fmt="%s")

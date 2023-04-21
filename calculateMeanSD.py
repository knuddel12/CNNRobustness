import matplotlib
matplotlib.use('Agg')
import numpy as np
from training import init_dataloaders, paths, init_datasets, reset_randomness
from Utilities.Datasets.transforms import Load, cache_preprocessing
from Dev.LarsDev.transforms import FaceAlignAndCrop, Grayscale
import tqdm
import matplotlib.pyplot as plt

""" Calculate the mean of every pixel over whole train data set to subtract 
    from pixel for zero-centering data cloud.
    Then, normalize each dimension (i.e. each pixel over whole data set (train set)) by finding 
    min/max value of a pixel and scaling it to -1 / +1.
    """
reset_randomness()
batchsize = 1
input_size = 112
csv_name = 'facescrub_cleanup_mini.csv'
CSV_PATH, CACHE_PATH, platform = paths(csv_name)

""" define transforms """
transforms_once = [Load(cache_images=True, cache_folder=CACHE_PATH),
                   FaceAlignAndCrop(),
                   #Grayscale(),
                   cache_preprocessing(CACHE_PATH)
                   ]
dataset_train, dataset_test, dataset_validation = init_datasets(CSV_PATH, transforms_once, image_only_transforms=None)

train_dataloader, test_dataloader, validation_dataloader = init_dataloaders(dataset_train,
                                                                            dataset_test,
                                                                            dataset_validation,
                                                                            batchsize,
                                                                            shuffle=False)

def meanFace():
    """ Calculates and saves a mean face image of the training set. Iterates through the training set and adds up every
    image by normalizing it with the number of iterations.
    """
    j = 1
    meanValues = 0
    images = 0
    for i, data in tqdm.tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader)):
        if i == 0:
            meanValues = data[0][0, :, :, :]
        else:
            j += 1
            meanValues = (meanValues * ((j - 1) / j)) + (data[0][0, :, :, :] * (1 / j))
            plt.imshow(meanValues / 255)
            plt.show()
    print(f"{j - 1} images have been stacked.")
    print(f"{meanValues.shape} is the dimension of the meanValues vector")
    plt.imshow(meanValues / 255)
    plt.savefig("meanFace_ausprobieren.png")
    print(f"File meanFace.png has been saved.")
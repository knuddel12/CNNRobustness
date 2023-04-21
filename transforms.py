import matplotlib
from Utilities.Datasets.transforms import log_preprocessing
matplotlib.use("agg")
import pandas as pd
from FaceSDK.Wrapper.DermalogFaceSDK.FaceDetection import FaceDetector
from FaceSDK.Wrapper.DermalogFaceSDK.FacePreprocessing import FacePreprocessor, PreprocessingType
from FaceSDK.Wrapper.DermalogFaceSDK.FaceDataExchange import FacePosition
from FaceSDK.Wrapper.HighLevelWrapper.util import load_image
import torch
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset

""" Transforms for Face Images fo the Face Classification """

class FaceAlignAndCrop(object):
    def __init__(self):
        self.fdet = FaceDetector()
        self.fpro = FacePreprocessor()

    @log_preprocessing()
    def __call__(self, sample):
        image, boundingbox = sample['computed']['image'], sample['boundingbox']
        boundingbox[0] = boundingbox[0] + boundingbox[2] / 2
        boundingbox[1] = boundingbox[1] + boundingbox[3] / 2
        image_sdk = load_image(image)
        face_position_sdk = FacePosition(boundingbox)
        points = self.fdet.find_face_points(image_sdk, face_position_sdk)
        cropped_image, _ = self.fpro.get_preprocessed_face(image_sdk, points, PreprocessingType.FACE_ALGORITHM_RESNET5_RGB)
        w, h = cropped_image.shape
        cropped_image_np = np.ascontiguousarray(np.frombuffer(cropped_image.get_buffer(),
                                                      dtype=np.uint8,
                                                      offset=54).reshape([h, w, 3])[::-1])
        sample['computed']['image'] = cropped_image_np
        return sample

class Load:
    def __init__(self, cache_images=False, cache_folder='D:/Datasets'):
        self._enable_caching = cache_images
        self._cache = cache_folder
        self.__dset_path_start_idx = None

class ToTensor:
    def __init__(self, device=None, keys=['image']):
        self.keys = keys
        if isinstance(device, str):
            device_idx = device.split(":")[-1]
        else:
            device_idx = device
        self.__device = torch.device('cpu') if device is None else torch.device(f'cuda:{device_idx}')

class cache_preprocessing:
    def __init__(self, folder, annotation_keys=None, lazyloading=True, read_only=False, caching_method='lmdb'):
        self.folder = folder
        self.annotation_keys = annotation_keys
        self.__lazyloading = lazyloading
        self.__read_only = read_only
        self.caching_method = caching_method

""" Unused """

class FaceBoundingboxDataset(Dataset):

  def __init__(self, csv_file, transform=None, target_transform=None):
    """
    Args:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
    self.faces_frame = pd.read_csv(csv_file)
    #self.root_dir = root_dir
    self.transform = transform
    self.target_transform = target_transform
    # filter out empty BoundingBox entries
    self.faces_frame = self.faces_frame.dropna()
    # filter out gray pictures
    for x in self.faces_frame.index:
        if self.faces_frame.loc[x, "modality"] == 'gray':
            self.faces_frame.drop(x, inplace=True)

  def __len__(self):
    return len(self.faces_frame)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    IDmap = list(set(self.faces_frame.iloc[:, 3]))
    img_path = self.faces_frame.iloc[idx, 1]
    image = io.imread(img_path)
    boundingbox = self.faces_frame.iloc[idx, 4:8]
    boundingbox = np.array([boundingbox])
    boundingbox = boundingbox.astype('int').reshape(4)
    person_id = IDmap.index(self.faces_frame.iloc[idx, 3])
    person_id = torch.tensor(person_id)

    image_id = self.faces_frame.iloc[idx, 0]
    sample = {'image': image, 'boundingbox': boundingbox}

    if self.transform:
        sample = self.transform(sample)
    if self.target_transform:
        person_id = self.target_transform(person_id)
    return sample['image'], person_id

class FaceCrop(object):
    def __call__(self, sample):
        image, boundingbox = sample['computed']['image'], sample['boundingbox']

        """  exception: if face boundingbox coordinates are out of image limits, zero-pad  """
        x = np.ones((3, 3))
        y = np.pad(x, ((0, 3), (3, 0)))

        """ 
        Padding variables for if boundingbox is out of image bounds, to keep original form of the boundingbox. 
        Are set to the amount of pixels to be padded. If boundingbox is not out of image, padding variables stay 0. """
        padding_x = list((0,0))
        padding_y = list((0,0))

        if boundingbox[0] < 0:
            face_x = 0
            padding_x[0] = -boundingbox[0]
        else:
            face_x = boundingbox[0]
        if (boundingbox[0]+boundingbox[2]) > len(image[0]):
            face_x2 = len(image[0]) - 1
            padding_x[1] = boundingbox[0]+boundingbox[2] - len(image[0])
        else:
            face_x2 = boundingbox[0] + boundingbox[2]
        if boundingbox[1] < 0:
            face_y = 0
            padding_y[0] = -boundingbox[1]
        else:
            face_y = boundingbox[1]
        if (boundingbox[1]+boundingbox[3]) > len(image):
            face_y2 = len(image)
            padding_y[1] = boundingbox[1]+boundingbox[3] - len(image)
        else:
            face_y2 = boundingbox[1] + boundingbox[3] - 1
        try:
            img = image[face_y:face_y2, face_x:face_x2]
        except TypeError:
            print('')
        img = np.pad(img, (padding_y, padding_x, (0,0)), constant_values=255)

        sample['computed']['image'] = img
        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If int, output is quadratic.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    def __call__(self, sample):
        image = sample['computed']['image']
        if isinstance(self.output_size, int):
            img = transform.resize(image, (self.output_size, self.output_size),preserve_range=True)
        else:
            img = transform.resize(image, (self.output_size[0], self.output_size[1]),preserve_range=True)
        sample['computed']['image'] = img
        return sample

class Grayscale(object):
    """Grayscale the rgb image. """
    @log_preprocessing()
    def __call__(self, sample):
        image = sample['computed']['image']
        image = np.dot(image, [0.2989, 0.5870, 0.1140])
        image = np.repeat(image[..., np.newaxis], 3, -1)
        sample['computed']['image'] = image
        return sample

class Zerocenter(object):
    """ Subtract the mean across every individual pixel.
     Centers data cloud around the origin along every dimension (number of pixels)."""
    def __call__(self, sample):
        means = np.loadtxt('meanValues1.txt')
        means = np.reshape(means, (112,112))
        means = np.stack((means, means, means))
        means = np.transpose(means, (1, 2, 0))
        image = sample['computed']['image']
        image = image - means
        sample['computed']['image'] = image
        return sample

class normalize(object):
    """ Subtract the mean across every individual pixel.
     Centers data cloud around the origin along every dimension (number of pixels)."""
    def __call__(self, sample):
        stddevs = np.loadtxt('stddeviations.txt')
        #means = np.reshape(means, (112,112))
        stddevs = np.stack((stddevs, stddevs, stddevs))                  #stacking to have dim [3, 112, 112] instead of [112, 112]
        stddevs = np.transpose(stddevs, (1, 2, 0))
        image = sample['computed']['image']
        image = image / stddevs
        sample['computed']['image'] = image*255             # factor 255 because image gets divided by 255 in following ToTensor() preprocessing step
        return sample

class meanLuminance(object):
    """to zero-center to the overall mean luminance, bring the mean luminance of the image to 255/2 = 127.5 """
    def __call__(self, sample):
        image = sample['computed']['image']
        image = ( image / np.mean(image) ) * 127.5
        sample['computed']['image'] = image
        return sample
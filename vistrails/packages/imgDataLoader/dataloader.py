"""DataLoader module"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from .dataset import ImageDataset
from keras.preprocessing.image import ImageDataGenerator
import cv2



class ImageDataLoader(object):
    """ImageDataLoader class. Define a image loader with capability of
    pre processing and data augmentation

    Args:
        dataset: ImageDaset
            A ImageDataset object.
        batch_size: int
            Number of images retrieved in each batch.
        fit_sample_size: int,
            Number of samples used to fit preprocessing and augmentation
            models.
        img_resize: tuple(int,int)
            Images will be resized for this size, the same size of net input
        img_preprocessing: dict {str: bool|float}
            A dictionary from conf.py (IMG_PREPROCESSING) containing
            configurations to preprocessing images.
        img_data_augmentation: dict {str: bool|float|int|None}
            A dictionary from conf.py (IMG_AUGMENTATION) containing
            configurations to augment images.
        preproc_and_augmentation: ImageDataGenerator
            ImageDataGenerator used for preprocessing parameter discovery. The
            same ImageDataGenerator must be used to fit train, val and test
    """

    def __init__(self,
                 dataset,
                 batch_size=32,
                 fit_sample_size=1000,
                 img_resize=None,
                 img_preprocessing=None,
                 img_data_augmentation=None,
                 preproc_and_augmentation=None):

        self._dataset = dataset
        self._batch_size = batch_size
        self._fit_sample_size = fit_sample_size
        self._img_resize = img_resize
        self._tot_num_images = len(self._dataset)

        self._positions = np.arange(self._tot_num_images)
        self._img_preprocessing = img_preprocessing
        self._img_data_augmentation = img_data_augmentation

        # create and fit data generator
        if preproc_and_augmentation is None:
            self._create_keras_data_generator()
            self._fit_generator()
        else:
            self._data_gen = preproc_and_augmentation

        print("DATAGEN: ", self._data_gen)

    def get_preproc_and_augmentation(self):
        """Return the image data generator for preproc and augment images"""
        return self._data_gen

    def _create_keras_data_generator(self):
        """ Create a keras ImageDataGenerator object"""

        # preproc is a must have field
        if self._img_preprocessing is None:
            raise ValueError(
                "IMG_PREPROCESSING field in conf.py must be defined")

        # preproc parameter
        preproc_params = {"featurewise_std_normalization": "bool",
                          "samplewise_center": "bool",
                          "samplewise_std_normalization": "bool",
                          "zca_whitening": "bool",
                          "zca_epsilon": "float",
                          "rescale": "float"}

        # check if all parameters are defined
        if not all([p in self._img_preprocessing for p in preproc_params]):
            params_str = " ".join("%s: %s\n" % (
                key, preproc_params[key]) for key in preproc_params)
            raise ValueError(
                """IMG_PREPROCESSING field in conf.py file must
                contains all features:\n %s""" % params_str)

        # image augmentation parameter
        if self._img_data_augmentation is not None:

            img_opt = {"rotation_range": "float",
                       "width_shift_range": "float",
                       "height_shift_range": "float",
                       "shear_range": "float",
                       "zoom_range": "float",
                       "channel_shift_range": "float",
                       "fill_mode": "str",
                       "cval": "float",
                       "horizontal_flip": "bool",
                       "vertical_flip": "bool"}
            # check if all parameters are defined
            if not all([p in self._img_data_augmentation for p in img_opt]):
                params_str = " ".join("%s: %s\n" % (
                    key, img_opt[key]) for key in img_opt)
                raise ValueError(
                    """IMG_AUGMENTATION field in conf.py file must contains all
                    features:\n %s""" % params_str)

        # if contains just preproc on dataloader
        if self._img_data_augmentation is None:
            preproc = self._img_preprocessing
            self._data_gen = ImageDataGenerator(
                featurewise_center=preproc['featurewise_center'],
                samplewise_center=preproc['samplewise_center'],
                featurewise_std_normalization=preproc[
                    'featurewise_std_normalization'],
                samplewise_std_normalization=preproc[
                    'samplewise_std_normalization'],
                zca_whitening=preproc['zca_whitening'],
                zca_epsilon=preproc['zca_epsilon'],
                rescale=preproc['rescale']
            )
        else:
            preproc = self._img_preprocessing
            img_aug = self._img_data_augmentation
            self._data_gen = ImageDataGenerator(
                featurewise_center=preproc['featurewise_center'],
                samplewise_center=preproc['samplewise_center'],
                featurewise_std_normalization=preproc[
                    'featurewise_std_normalization'],
                samplewise_std_normalization=preproc[
                    'samplewise_std_normalization'],
                zca_whitening=preproc['zca_whitening'],
                zca_epsilon=preproc['zca_epsilon'],
                rescale=preproc['rescale'],
                rotation_range=img_aug['rotation_range'],
                width_shift_range=img_aug['width_shift_range'],
                height_shift_range=img_aug['height_shift_range'],
                shear_range=img_aug['shear_range'],
                zoom_range=img_aug['zoom_range'],
                channel_shift_range=img_aug['channel_shift_range'],
                fill_mode=img_aug['fill_mode'],
                cval=img_aug['cval'],
                horizontal_flip=img_aug['horizontal_flip'],
                vertical_flip=img_aug['vertical_flip'])

    def _fit_generator(self, dataset=None):
        """Calculate the parameter for pre processing and data augmentation"""

        # if dataset is not use the self image set
        # otherwise use the defined dataset to fit preproc parameter
        # the preproc for validation and test must have same parameter
        # from trainning passing trainning dataset to fit the
        # right values can be defined
        if dataset is None:
            dataset = self._dataset

        positions = np.arange(len(dataset))

        # shuffle positions arrays
        images = []
        np.random.shuffle(positions)
        for position in positions[:self._fit_sample_size]:
            img, _ = dataset[position]
            images.append(img)

        images = np.array(images)
        self._data_gen.fit(images)

    def get_batch(self):
        """Return a batch of images

        Returns:
            np.array
                A numpy array of images
            np.array
                A numpy array with vectorized labels
        """

        while True:
            images = []
            labels = []

            np.random.shuffle(self._positions)
            for posi in self._positions[:self._batch_size]:
                img, label = self._dataset[posi]

                if self._img_resize is not None:
                    img = cv2.resize(img, self._img_resize,
                                     interpolation=cv2.INTER_CUBIC)

                images.append(img)
                labels.append(label)

            images = np.array(images)

            labels = np.array(labels)

            yield self._data_gen.flow(images, labels, self._batch_size).next()


def get_train_data_generator(conf):
    """ Return a generator for train data

    Args:
        conf: dict
            A dictionary from conf.py

    Returns:
        ImageDataLoader object
    """
    train_file_path = conf.IMG_DATASET['train']
    hot_encode_labels = conf.IMG_DATASET['hot_encode_labels']

    # create image dataset
    img_dataset = ImageDataset(train_file_path, hot_encode_labels)

    # image size (inputs will be resized for network input)
    img_size = conf.CONV_NET_ARCHITECTURE['img_input_size'][:-1]

    # get preproc configs
    preproc = conf.IMG_PREPROCESSING

    # get augmentation configs
    augmentation = conf.IMG_AUGMENTATION

    # fit loader size
    fit_size = conf.DATA_LOADER['fit_sample_size']

    # train_batch size
    batch_size = conf.DATA_LOADER['train_batch_size']

    # create dataloader
    data_loader = ImageDataLoader(img_dataset,
                                  batch_size=batch_size,
                                  fit_sample_size=fit_size,
                                  img_resize=img_size,
                                  img_preprocessing=preproc,
                                  img_data_augmentation=augmentation)

    return data_loader


def get_val_data_generator(conf, preproc_and_augmentation=None):
    """ Return a generator for val data

        Args:
            conf: dict
                A dictionary from conf.py

        Returns:
            ImageDataLoader object
    """
    val_file_path = conf.IMG_DATASET['validation']
    hot_encode_labels = conf.IMG_DATASET['hot_encode_labels']

    # create image dataset
    img_dataset = ImageDataset(val_file_path, hot_encode_labels)

    # image size (inputs will be resized for network input)
    img_size = conf.CONV_NET_ARCHITECTURE['img_input_size'][:-1]

    # get preproc configs
    preproc = conf.IMG_PREPROCESSING

    # fit loader size
    fit_size = conf.DATA_LOADER['fit_sample_size']

    # train_batch size
    batch_size = conf.DATA_LOADER['validation_batch_size']

    # create dataloader
    data_loader = ImageDataLoader(
        img_dataset,
        batch_size=batch_size,
        fit_sample_size=fit_size,
        img_resize=img_size,
        img_preprocessing=preproc,
        img_data_augmentation=None,
        preproc_and_augmentation=preproc_and_augmentation
    )

    return data_loader


def get_test_data_generator(conf, preproc_and_augmentation=None):
    """ Return a generator for test data

        Args:
            conf: dict
                A dictionary from conf.py

        Returns:
            ImageDataLoader object
    """
    test_file_path = conf.IMG_DATASET['test']
    hot_encode_labels = conf.IMG_DATASET['hot_encode_labels']

    # create image dataset
    img_dataset = ImageDataset(test_file_path, hot_encode_labels)

    # image size (inputs will be resized for network input)
    img_size = conf.CONV_NET_ARCHITECTURE['img_input_size'][:-1]

    # get preproc configs
    preproc = conf.IMG_PREPROCESSING

    # fit loader size
    fit_size = conf.DATA_LOADER['fit_sample_size']

    # train_batch size
    batch_size = conf.DATA_LOADER['test_batch_size']

    # create dataloader
    data_loader = ImageDataLoader(
        img_dataset,
        batch_size=batch_size,
        fit_sample_size=fit_size,
        img_resize=img_size,
        img_preprocessing=preproc,
        img_data_augmentation=None,
        preproc_and_augmentation=preproc_and_augmentation
    )

    return data_loader

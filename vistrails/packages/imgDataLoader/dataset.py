# -*- coding: utf-8 -*-

'''
    Dataset module.
'''

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import cv2


class ImageDataset(object):
    """ ImageDataset class. Define a way to load images from files and access
    them per index.
        Args:
            imgs_file: str
                CSV file associating image path with a class
                name (str) or with a vector. The first position
                on each row in csv must be the image path. The
                following data will be  either a string with
                the name of the image class (classification)
                or a vector of floats. The delimiter of the
                file must be semicolon.

            hot_encode_labels: bool
                A boolean flag. If True dataset will transform each string
                label associated with the image into a hot encoded vector.
                If False it is spected vectorized label into imgs_file
        """

    def __init__(self,
                 imgs_file,
                 hot_encode_labels=False):

        self._imgs_file = imgs_file
        self._hot_encode_labels = hot_encode_labels

        self._labels = []
        self._imgs_path = []
        self._labels_position = []
        self._dataset_size = 0

        self._load_dataset_metadata()

    def _load_dataset_metadata(self):
        """ Load the metada about image dataset storing data into
            specific obj variables."""

        # check if imgs_file exists
        if not os.path.isfile(self._imgs_file):
            raise ValueError("File %s doesn't exists." % (self._imgs_file))

        # loading img  description file
        with open(self._imgs_file) as label_file_content:
            for line in label_file_content:
                data = line.strip().split(";")

                # load img path
                self._imgs_path.append(data[0])

                # labels will be vectorized after initial load
                if self._hot_encode_labels:
                    self._labels.append(data[1])
                else:
                    # transforming raw data to a list of float
                    self._labels.append(
                        np.array([float(el) for el in data[1:]]))

        # if necessary, hot encode vectors
        if self._hot_encode_labels:
            self._labels_position = list(set(self._labels))
            self._labels_position.sort()

            num_labels = len(self._labels_position)
            num_samples = len(self._labels)

            for i in range(num_samples):
                label = self._labels[i]
                one_position = self._labels_position.index(label)
                hot_enc_vector = np.zeros(num_labels)
                hot_enc_vector[one_position] = 1.0
                self._labels[i] = hot_enc_vector

        # get dataset size
        self._dataset_size = len(self._imgs_path)

    def __len__(self):
        """The total of images that can be loaded """
        return self._dataset_size

    def __getitem__(self, idx):
        """Get a specific image with associated label
            idx: int
                A valid index position for an image in dataset.

            Return:
                A dictionary with a unique sample wit_labelsh label and
                image as np.array
                {"label": "image_label, "img": np.array}
        """

        img_path = self._imgs_path[idx]

        # loading image
        img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)

        # load label
        label = self._labels[idx]

        return img, label

    def get_class_labels(self):
        """List the class names
        Return:
            list of str: a list with the name of each class
        """
        return self._labels_position

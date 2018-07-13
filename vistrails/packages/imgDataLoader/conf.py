# -*- coding: utf-8 -*-
'''
    Configuration file
'''
from datetime import datetime

# Preprocessing methods applied in each image (train or classification)
IMG_PREPROCESSING = {
    "featurewise_center": False,
    "featurewise_std_normalization": True,
    "samplewise_center": False,
    "samplewise_std_normalization": False,
    "zca_whitening": False,
    "zca_epsilon": 1e-6,
    "rescale": 1/255.0
}

# Methods of data augmentation used only on training
IMG_AUGMENTATION = {
    "rotation_range": 0,
    "width_shift_range": 0.3,
    "height_shift_range": 0.3,
    "shear_range": 0.1,
    "zoom_range": 0.0,
    "channel_shift_range": 0.,
    "fill_mode": 'nearest',
    "cval": 0.,
    "horizontal_flip": True,
    "vertical_flip": False
}

# Define the maximo size of memory used to store images and batch size
DATA_LOADER = {
    "fit_sample_size": 10000,
    "train_batch_size": 256,
    "validation_batch_size": 256,
    "test_batch_size": 8
}

# ConvNet architecture parameter
CONV_NET_ARCHITECTURE = {
    "architecture": "resnet50",
    "img_input_size": (64, 64, 3),  # w, h, c
    "output_size": 10,
    "softmax_output": True,
    "conv2d_initializer": "glorot_uniform"
}

# Train parameters
CONV_NET_TRAIN = {
    "steps_per_epoch": 16,
    "epochs": 1000,
    "early_stop": {
        "monitor": 'val_loss',
        "patience": 70,
        "min_delta": 0.00001
    },

    "loss": "categorical_crossentropy",
    "metrics": ["accuracy"]
}

# Optimizer configurations used to training the network
OPTIMIZER = {
    "method": "adam",
    "params": {
        "lr": 0.001,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-8,
        "decay": 1e-4}
}

# Folders with images and labelfiles
IMG_DATASET = {
    "hot_encode_labels": True,
    "train": "/home/motumbo/Desktop/fabricio/dataset/cifar-100/train-filtered.csv",
    "validation": "/home/motumbo/Desktop/fabricio/dataset/cifar-100/test-filtered.csv",
    "test": "/home/motumbo/Desktop/fabricio/dataset/cifar-100/test-filtered.csv"
}

# Folder where experiments will be stored
EXPERIMENT_PATH = "/home/motumbo/Desktop/fabricio/experiment/" \
                  + datetime.now().strftime("%Y-%m-%d_%H:%M")

import importlib
import os
import signal

import numpy as np

from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator


def safe_create_dir(dir):
    """Safely create the given directory."""
    if not os.path.exists(dir):
        os.mkdir(dir)


def preprocess_image(x):
    """Scale image array to have all pixels be between -1 and 1.

    Args:
        x: A numpy array representing an image.

    Returns:
        A numpy array with the same shape as x but with all values
        scaled to be between -1 and 1.
    """
    return x / 255 * 2 - 1


def load_image(args):
    """Load an image file and conduct all preprocessing.

    Args:
        args: A 3 tuple of (config, image_filename, augment) where
        image_data_config is an ImageDataConfig static class,
        image_filename is the image that will be loaded,
        and augment is a boolean stating whether data augmentation should
        occur using augmentation parameters in the configuration.

    Returns: A numpy array representing the preprocessed image.
    """
    image_data_config, image_filename, augment = args
    image = load_img(image_filename)
    image_array = img_to_array(image)
    
    # Update: normalize the image on the fly
    image_array = preprocess_image(image_array)

    if augment:
        augmenter = ImageDataGenerator(**image_data_config.augment_params)
        image_array = augmenter.random_transform(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class ParallelModelCheckpoint(ModelCheckpoint):

    def __init__(self, model, *args, **kwargs):
        """Instantiate the ParallelModelCheckpoint.

        Set model here so we can pass in a model allocated in CPU memory, rather
        than the multi-GPU replicated model.

        Args:
            model: A keras model we want to save checkpoints for. Should be allocated on
                   a single device (either GPU or CPU).
        """
        super(ParallelModelCheckpoint, self).__init__(*args, **kwargs)
        self.model = model

    def set_model(self, model):
        """Skip setting model so that we can refer to the model passed in at init time.

        Useful for multi-GPU training.

        Args:
            model: A keras model that will be passed in when it's `fit` method is called. When
                   using multi-GPU training, this needs to be ignored, so we won't do anything.
        """
        pass


def load_model_config(config_name):
    """Load a model configuration.

    Model configurations include a `build_model` function that accepts input shape tuples as input.
    As an example, if a model only accepts an image as input, it will accept a single input shape
    tuple.

    The configuration is responsible for setting up all output tensors, and providing a `build_labels` function
    for extracting label data from the raw data csv file provided by MGH.

    Args:
        config_name: A string denoting which configuration to load

    Returns:
        A loaded module implementing the API described in this doc comment.
    """
    return importlib.import_module('models.{}'.format(config_name))

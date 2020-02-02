import cPickle as pickle
import os

import click
import numpy as np
import pandas as pd

from collections import namedtuple
from itertools import izip

from PIL import Image

from keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split

from util import safe_create_dir, load_model_config


FileData = namedtuple('FileData', ['image_id', 'original_filename', 'original_file_path'])
ProcessedImageMetadata = namedtuple('ProcessedImageMetadata',
                                    ['image_id', 'image_filename', 'image_label'])


def load_label_data(config):
    """Load labels and associate label with normalized patient ID.

    Splits into training and validation sets, stratified by label.

    Args:
        config: A model configuration module.

    Returns: Two dicts mapping image_ids to label for training and validation.
    """
    label_data = pd.read_csv(config.LabelDataConfig.data_path)
    ids = list(label_data['Training cases final'])
    labels = config.build_labels(label_data)

    train_ids, val_ids, train_labels, val_labels = train_test_split(
        ids,
        labels,
        stratify=labels,
        train_size=config.ImageDataConfig.train_percent)

    train_label_data = {image_id.upper(): label
                        for image_id, label in izip(train_ids, train_labels)}
    val_label_data = {image_id.upper(): label
                      for image_id, label in izip(val_ids, val_labels)}

    return train_label_data, val_label_data


def load_filename_data(config):
    """Loads filenames for all images as well as the normalized ID associated with the image.

    Args:
        config: A model configuration module.

    Returns: A list of FileData namedtuple.
    """
    image_filename_data = []
    image_filenames = os.listdir(config.ImageDataConfig.data_path)
    for image_filename in image_filenames:
        # changing below to account for our naming convention of chunks
	image_id= image_filename.split('_')[0]
        image_id = image_id.upper()
        original_file_path = os.path.join(config.ImageDataConfig.data_path, image_filename)
        image_filename_data.append(FileData(image_id, image_filename, original_file_path))

    return image_filename_data

# Given that we've already preprocessed our images, we no longer have a need to re-save (or rename)
# our images as below.  We do need to aggregate the metadata to pass at training time. So, we'll
# modify the code below to use the original_images directory instead of the preprocessed_images
# one.  We'll just comment out the lines that re-save the images, and we'll use the original
# filepath instead of a new one.
# We'll also need to modify the model config files so that ImageDataConfig.preprocessed_image_path
# points to data/original_images.  This is hacky but simplest for now, and gives us flexibility in
# case we want to go back to the original preprocessing scheme later.
def preprocess_images_and_labels(config,
                                 image_file_data,
                                 train_image_id_labels,
                                 val_image_id_labels):
    """Read and preprocess all images, record metadata for new image locations, labels, and IDs.
    Args:
        config: A model configuration module.
        image_file_data: A list of FileData namedtuples to be preprocessed and copied.
        train_image_id_labels: A dict mapping training image_ids to labels.
        val_image_id_labels: A dict mapping validaiton image_ids to labels.

    Returns: A list of ProcessedImageMetadata namedtuples.
    """
    # Commenting out the line below in keeping with the comment block above the function. We don't
    # actually need to do this, because safe_create_dir will never overwrite an existing directory,
    # but better to be safe.
    # safe_create_dir(config.ImageDataConfig.preprocessed_image_path)

    # Add in a counter for tracking progress via the console
    counter = 0

    train_image_metadata, val_image_metadata = [], []
    for image_data in image_file_data:

        if image_data.image_id in train_image_id_labels:
            image_metadata = train_image_metadata
            image_label = train_image_id_labels[image_data.image_id]
        else:
            image_metadata = val_image_metadata
            image_label = val_image_id_labels[image_data.image_id]

        # TODO Stop squishing the image and handle cropping correctly sized windows at sample time.

        # Comment out the chunk below to avoid re-saving our images, which we have
        # already preprocessed.  We just want to generate metadata for them.
	"""
        image = load_img(
            image_data.original_file_path,
            target_size=config.ImageDataConfig.size)
        new_file_path = os.path.join(
            config.ImageDataConfig.preprocessed_image_path,
            image_data.original_filename.upper().replace('PNG', 'JPG'))  # Convert all images to jpegs.
        image.save(new_file_path, format='JPEG', quality=85)
	"""

        # We generate metadata, setting the image filepath as the original filepath, as we
        # have already preprocessed beforehand.
        original_file_path = image_data.original_file_path

        image_metadata.append(ProcessedImageMetadata(image_data.image_id, original_file_path, image_label))

    return train_image_metadata, val_image_metadata


def write_metadata(config, train_image_metadata, val_image_metadata):
    """Write the preprocessed image metadata to disk.

    Args:
        config: A model configuration module.
        train_image_metadata: A list of ProcessedImageMetadata namedtuples for training.
        val_image_metadata: A list of ProcessedImageMetadata namedtuples for validation.
    """
    with open(config.ImageDataConfig.preprocessed_image_metadata_filename, 'wb') as f:
        pickle.dump({'train': train_image_metadata, 'val': val_image_metadata}, f, pickle.HIGHEST_PROTOCOL)


@click.command()
@click.option('--config_name')
def main(config_name):
    config = load_model_config(config_name)
    train_image_id_labels, val_image_id_lables = load_label_data(config)
    image_file_data = load_filename_data(config)

    train_image_metadata, val_image_metadata = preprocess_images_and_labels(
        config,
        image_file_data,
        train_image_id_labels,
        val_image_id_lables)

    write_metadata(config, train_image_metadata, val_image_metadata)


if __name__ == '__main__':
    main()

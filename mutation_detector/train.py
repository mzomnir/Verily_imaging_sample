# -*- coding: utf-8 -*-
import cPickle as pickle

from multiprocessing import Pool

import click
import keras.backend as K
import numpy as np

from keras.callbacks import Callback
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
from keras.utils import multi_gpu_model
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from util import load_image, ParallelModelCheckpoint, init_worker, load_model_config
from preprocess_data import ProcessedImageMetadata

def labeled_data_generator(config, X, y, augment=False):
    """A generator function that yields images and labels ready to pass into `Model.fit_generator`.

    Args:
        config: A configuration module that implements a `build_model` function.
        X: A numpy array of image filenames to be sampled and read.
        y: A numpy array of image labels.
        augment: A boolean stating if images should be augmented.

    Returns: A generator that iterates indefinitely over the given data.
    """
    pool = Pool(config.TrainingConfig.n_pipeline_workers, initializer=init_worker)

    while True:
        sampled_indices = np.random.choice(np.arange(len(y)), size=config.TrainingConfig.batch_size)
        X_b, y_b = X[sampled_indices], y[sampled_indices]
        load_image_args = []
        for image_filename in X_b:
            load_image_args.append((config.ImageDataConfig, image_filename, augment))
        # Note: per the update to load_image() in util.py, the load_image() function
        # now normalizes the numpy array before returning
	X_b_images = np.vstack(pool.map(load_image, load_image_args))
        yield X_b_images, y_b


def load_labeled_data(config):
    """Read preprocessed image metadata.

    Args:
        config: A configuration module that implements a `build_model` function.

    Returns: A 4-tuple of (X_train, X_val, y_train, y_val)
    """
    with open(config.ImageDataConfig.preprocessed_image_metadata_filename, 'rb') as f:
        train_val_image_metadata = pickle.load(f)

    def _prep_labeled_metadata(image_metadata):
        X = np.array([example.image_filename for example in image_metadata])
        y = np.vstack([example.image_label for example in image_metadata])
        n_labels = len(y[0])
        y = y.reshape(-1, n_labels)
        return X, y

    X_train, y_train = _prep_labeled_metadata(train_val_image_metadata['train'])
    X_val, y_val = _prep_labeled_metadata(train_val_image_metadata['val'])

    return X_train, X_val, y_train, y_val

# Define a class to record and print F1, precision, and recall after each epoch
class Metrics(Callback):

    def __init__(self, val_data_generator, val_steps):
	self.val_data_generator = val_data_generator
	self.val_steps = val_steps

    def on_train_begin(self, logs={}):
        
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
 

    def on_epoch_end(self, epoch, logs={}):
        
	val_labels = []
	val_predictions = []

	for i in range(self.val_steps):
	    image, label = next(self.val_data_generator)
	    val_labels.extend(label)
	    val_predictions.extend(np.round(self.model.predict(image)))

        val_targ = np.vstack(val_labels)
	val_predict = np.vstack(val_predictions)

        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        
        # self.val_f1s.append(_val_f1)
        # self.val_recalls.append(_val_recall) 
        # self.val_precisions.append(_val_precision)
        
        print ' — val_f1: {} — val_precision: {} — val_recall {}'.format(_val_f1, _val_precision, _val_recall)
        return
 

def setup_model(config, n_gpus=None):
    """Build the model using the provided configuration.

    Args:
        config: A configuration module that implements a `build_model` function.
        n_gpus: An int, the number of GPUs to be used for training.

    Returns: A Keras Model object that is ready to train.
    """
    if n_gpus is None or n_gpus == 1:
        # TODO use the base_model to set up callbacks
        base_model = model = config.build_model(config.ImageDataConfig.shape)
    else:
        # multi-GPU training requires first allocating the model in CPU memory, then
        # setting it up for multi-GPU training.
        with K.tf.device('/cpu:0'):
            base_model = config.build_model(config.ImageDataConfig.shape)

        model = multi_gpu_model(base_model, gpus=n_gpus)

    model.compile(
        optimizer=config.setup_optimizer(),
        loss=config.setup_objective(),
        metrics=['accuracy'])

    # TODO return model checkpoint callbacks here as well.
    return model


@click.command()
@click.option('--config_name')
@click.option('--n_gpus', default=None, type=int)
def main(config_name, n_gpus):
    config = load_model_config(config_name)

    X_train, X_val, y_train, y_val = load_labeled_data(config)
    train_gen = labeled_data_generator(config, X_train, y_train, augment=False)
    val_gen = labeled_data_generator(config, X_val, y_val)

    model = setup_model(config, n_gpus)
   
    # checkpoint the weights for the best model so far, using accuracy
    val_acc_filepath="mutation_detector/checkpoints/val_acc_weights_round6.best.hdf5"
    # create a ModelCheckpoint to record model accuracy
    val_acc_checkpoint = ModelCheckpoint(val_acc_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    # logger to log the training performance
    csv_logger = CSVLogger('mutation_detector/logs/log_round6.csv', append=True, separator=';')

    # instantiate our custom callback that provides f1, precision, and recall for each epoch
    f1_precision_recall = Metrics(val_gen, config.TrainingConfig.val_steps)

    # populate the callbacks list with our model checkpoint, metrics, and logger
    callbacks_list = [val_acc_checkpoint, f1_precision_recall, csv_logger] 

    model.fit_generator(
        train_gen,
        steps_per_epoch=config.TrainingConfig.steps_per_epoch,
        epochs=config.TrainingConfig.epochs,
        validation_data=val_gen,
        validation_steps=config.TrainingConfig.val_steps,
        callbacks=callbacks_list,
        class_weight = [.3,.7])


if __name__ == '__main__':
    main()

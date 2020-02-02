import numpy as np

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import (Input, Conv2D, BatchNormalization, Activation,
                          Dense, MaxPooling2D, GlobalAveragePooling2D, Dropout)
from keras.models import Model
from keras.optimizers import RMSprop
from keras.regularizers import l2


# EXPERIMENT CONFIGURATION

class LabelDataConfig:
    data_path = 'data/Training_ML_clean.csv'


class ImageDataConfig:
    data_path = 'data/original_images'
    # generate metadata but do not alter the images themselves (we already did that)
    # preprocessed_image_path = 'data/preprocessed_images'
    preprocessed_image_path = 'data/original_images'
    preprocessed_image_metadata_filename = 'data/preprocessed_image_metadata.pkl'
    train_percent = 0.8
    augment_params = dict(horizontal_flip=True,
                          vertical_flip=True,
                          rotation_range=180,
                          zoom_range=0.2)
    size = (299, 299)
    shape = size + (3,)


class TrainingConfig:
    n_pipeline_workers = 10
    steps_per_epoch = 400
    val_steps = 200
    epochs = 25
    batch_size = 256
    n_gpus = 2


def setup_optimizer():
    return RMSprop(lr=0.0005)


def setup_objective():
    return 'binary_crossentropy'


# MODEL CONFIG


def build_labels(df):
    # Updating label for the EGFR-vs-rest classifier
    labels = list(df['EGFR_label'])
    labels = np.vstack([np.array(label).reshape(1, 1) for label in labels]).astype(np.float32)
    return labels


def conv_bn(x,
            n_filters,
            filter_size,
            strides=1,
            activation='relu',
            padding='same',
            # Adding in l2 regularization
	    kernel_regularizer=l2(0.1)):
    x = Conv2D(n_filters, filter_size, strides=strides, padding=padding)(x)
    # axis=3 assumes we'll be using the Tensorflow backend
    x = BatchNormalization(axis=3)(x)
    x = Activation(activation)(x)
    return x


def build_model(image_shape):
    image = Input(shape=image_shape)
    base_model = InceptionResNetV2(input_tensor=image, include_top=False)
    for layer in base_model.layers[:530]:
        layer.trainable = False
        if type(layer) == 'BatchNormalization':
            layer.momentum = 1.0

    h = conv_bn(base_model.layers[530].output, 128, filter_size=(3, 3))
    h = Dropout(0.1)(h)
    h = conv_bn(h, 128, filter_size=(3, 3))
    h = Dropout(0.1)(h)
    image_features = GlobalAveragePooling2D()(h)
    image_features_dropout = Dropout(0.2)(image_features)

    prediction = Dense(1, activation='sigmoid')(image_features_dropout)

    model = Model(inputs=image, outputs=prediction)

    return model

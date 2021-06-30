from tensorflow.keras import callbacks
from predict import pred_ae, init_kmeans
import random
import os
import config as cfg
import pandas as pd
from keras.models import Model
from sklearn.cluster import KMeans
from nets import ClusteringLayer, autoencoder
from metrics import nmi, ari, acc
from build_and_save_features import load_dataset
import nets
from time import time
import visualization as viz
from tensorflow.keras.optimizers import Adam
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator
import numpy as np
from generators import generator, MyImageGenerator
from keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam



def finetune_encoder(x_train_shape, x_val_shape, train_generator, val_generator, batch_size):
    
    _, encoder = nets.autoencoder()
    encoder.load_weights(os.path.join(cfg.models, 'aspc_26_Dense', 'ae', 'ce_weights'))
    
    for f in encoder.layers[:-1]:
        f.trainable = False
    
    encoder.summary()
    encoder.compile(
        optimizer=Adam(1e-5),
        loss='mse'
    )

    encoder.fit(
        train_generator,
        steps_per_epoch=math.ceil(x_train_shape / batch_size),
        epochs=1000,
        validation_data=val_generator,
        validation_steps=math.ceil(x_val_shape / batch_size),
        callbacks=cfg.my_callbacks
    )

if __name__ == "__main__":
    x_train, y_train = load_dataset('x_train.npy', 'y_train.npy')
    x_val, y_val = load_dataset('x_val.npy', 'y_val.npy')
    x_test, y_test = load_dataset('x_test.npy', 'y_test.npy')

    x_train = x_train.reshape(x_train.shape[0], 128, 128, 1)
    x_val = x_val.reshape(x_val.shape[0], 128, 128, 1)
    x_test = x_test.reshape(x_test.shape[0], 128, 128, 1)

    train_datagen = MyImageGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True, 
    )
    val_datagen = MyImageGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
    )

    # fit the data augmentation
    train_datagen.fit(x_train)
    val_datagen.fit(x_val)

    batch_size = 32

    # define generators
    train_generator = generator(
        image_generator=train_datagen,
        x=x_train,
        batch_size=batch_size
    )
    val_generator = generator(
        image_generator=val_datagen,
        x=x_val,
        batch_size=batch_size,
        shuffle=False
    )

    # finetuning
    finetune_encoder(
        x_train.shape[0],
        x_val.shape[0],
        train_generator, 
        val_generator,
        batch_size
    )

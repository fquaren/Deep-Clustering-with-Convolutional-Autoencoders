from tensorflow.keras import callbacks
from predict import pred_ae, init_kmeans
import random
import os
import config as cfg
import pandas as pd
from keras.models import Model
from sklearn.cluster import KMeans
from nets import autoencoder
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


def generators(x_train, x_val, batch_size):
    
    # define data augmentation configuration
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
    
    # generators
    train_generator = generator(train_datagen, x_train, batch_size=batch_size)
    val_generator = generator(val_datagen, x_val, batch_size=batch_size, shuffle=False)

    return train_generator, val_generator


def pretrain(
    autoencoder,
    encoder,
    x_train,
    train_generator,
    x_val,
    val_generator,
    batch_size,
    pretrain_epochs,
    my_callbacks,
    ae_models,
    optim
):

    print('pretraining...')
    autoencoder.summary()
    autoencoder.compile(optimizer=optim, loss='mse')

    # train model
    t0 = time()
    autoencoder.fit(
        train_generator,
        steps_per_epoch=math.ceil(x_train.shape[0] / batch_size),
        epochs=pretrain_epochs,
        validation_data=val_generator,
        validation_steps=math.ceil(x_val.shape[0] / batch_size),
        callbacks=my_callbacks
    ) 

    print('pretraining time: ', time() - t0)
    autoencoder.save_weights(os.path.join(ae_models, 'ae_weights'))
    encoder.save_weights(os.path.join(ae_models, 'ce_weights'))
    # save plot metrics
    cfg.d_ae['train_loss'] = autoencoder.history.history['loss']
    cfg.d_ae['val_loss'] = autoencoder.history.history['val_loss']
    print('weigths and metrics saved.')


if __name__ == "__main__":
    # get datasets
    x_train, y_train = load_dataset('x_train.npy', 'y_train.npy')
    x_val, y_val = load_dataset('x_val.npy', 'y_val.npy')
    x_test, y_test = load_dataset('x_test.npy', 'y_test.npy')


    os.makedirs(os.path.join(cfg.experiments, cfg.exp), exist_ok=True)
    os.makedirs(os.path.join(cfg.tables, cfg.exp), exist_ok=True)
    os.makedirs(os.path.join(cfg.figures, cfg.exp, 'ae'), exist_ok=True)
    os.makedirs(os.path.join(cfg.models, cfg.exp, 'ae'), exist_ok=True)

    autoencoder, encoder = nets.autoencoder()
    x_train = x_train.reshape(x_train.shape[0], 128, 128, 1)
    x_val = x_val.reshape(x_val.shape[0], 128, 128, 1)
    x_test = x_test.reshape(x_test.shape[0], 128, 128, 1)

    train_generator, val_generator = generators(
        x_train, 
        x_val, 
        cfg.ae_batch_size
    )

    # pretrain(
    #     autoencoder=autoencoder,
    #     encoder=encoder,
    #     x_train=x_train,
    #     train_generator=train_generator,
    #     x_val=x_val,
    #     val_generator=val_generator,
    #     batch_size=cfg.ae_batch_size,
    #     pretrain_epochs=cfg.pretrain_epochs,
    #     my_callbacks=cfg.my_callbacks,
    #     ae_models=cfg.ae_models,
    #     optim=cfg.ae_optim
    # )

    # _, _ = init_kmeans(
    #     n_clusters=cfg.n_clusters,
    #     n_init_kmeans=cfg.n_init_kmeans,
    #     x=x_train,
    #     y=y_train,
    # )


    df = pd.DataFrame(data=cfg.d_ae)
    df.to_csv(
        os.path.join(
            cfg.tables,
            cfg.exp,
            'ae_train.csv'
            ), 
        index=False
    )
    print('metrics saved.')

    viz.plot_pretrain_metrics(
        file=os.path.join(cfg.tables, cfg.exp, 'ae_train.csv'),
        save_dir=os.path.join(cfg.figures, cfg.exp, 'ae'),
    )
    print('plotted pretrained metrics.')
    
    viz.plot_ae_tsne(
        nets.encoder(), 
        cfg.ce_weights, 
        os.path.join(cfg.figures, cfg.exp, 'ae'), 
        x_test
    )
    viz.plot_ae_umap(
        nets.encoder(), 
        cfg.ce_weights, 
        os.path.join(cfg.figures, cfg.exp, 'ae'), 
        x_test
    )
    print('plotted kmeans.')

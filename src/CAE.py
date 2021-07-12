from predict import pred_ae
import random
import os
import config as cfg
import pandas as pd
from keras.models import Model
from sklearn.cluster import KMeans
from nets import ClusteringLayer
from metrics import nmi, ari, acc
from build_and_save_features import load_dataset
import predict
import numpy as np


def pretrainCAE(
        model,
        x_train,
        x_val,
        batch_size,
        pretrain_epochs,
        my_callbacks,
        cae_models,
        optim
            ):
    autoencoder, encoder = model
    encoder.summary()
    autoencoder.summary()
    autoencoder.compile(optimizer=optim, loss='mse')
    autoencoder.fit(
        x_train,
        x_train,
        batch_size=batch_size,
        epochs=pretrain_epochs,
        validation_data=(x_val, x_val),
        callbacks=my_callbacks
    )
    # save plot metrics
    cfg.dict_metrics['train_loss'] = autoencoder.history.history['loss']
    cfg.dict_metrics['val_loss'] = autoencoder.history.history['val_loss']

    pred_ae(
        net=autoencoder,
        weights=cfg.cae_weights,
        directory=cfg.train_directory,
    )


if __name__ == "__main__":
    # get datasets
    x_train, y_train = load_dataset('x_train.npy', 'y_train.npy')
    x_val, y_val = load_dataset('x_val.npy', 'y_val.npy')
    x_test, y_test = load_dataset('x_test.npy', 'y_test.npy')

    os.makedirs(os.path.join(cfg.experiments, cfg.exp), exist_ok=True)
    os.makedirs(os.path.join(cfg.tables, cfg.exp), exist_ok=True)
    os.makedirs(os.path.join(cfg.figures, cfg.exp, 'cae'), exist_ok=True)
    os.makedirs(os.path.join(cfg.models, cfg.exp, 'cae'), exist_ok=True)

    x_train = x_train.reshape(x_train.shape[0], 128, 128, 1)
    x_val = x_val.reshape(x_val.shape[0], 128, 128, 1)
    x_test = x_test.reshape(x_test.shape[0], 128, 128, 1)

    # pretrain CAE
    pretrainCAE(
        model=cfg.cae,
        x_train=x_train,
        x_val=x_val,
        batch_size=cfg.cae_batch_size,
        pretrain_epochs=cfg.pretrain_epochs,
        my_callbacks=cfg.my_callbacks,
        cae_models=cfg.cae_models,
        optim=cfg.cae_optim
    )

    predict.init_kmeans(
        x=x_train,
        x_val=x_test,
        y=y_train,
        y_val=y_test,
        random_state=None,
        weights=cfg.cae_weights,
    )

    # test_acc_list = []
    # test_nmi_list = []
    # for i in range(25):
    #     _, _, _, test_acc, test_nmi = predict.init_kmeans(
    #         x=x_train,
    #         x_val=x_test,
    #         y=y_train,
    #         y_val=y_test,
    #         random_state=None,
    #         weights=cfg.cae_weights,
    #     )
    #     test_acc_list.append(test_acc)
    #     test_nmi_list.append(test_nmi)

    # print('MEAN ACC:', np.mean(test_acc_list))
    # print('STD ACC:', np.std(test_acc_list))
    # print('MEAN NMI:', np.mean(test_nmi_list))
    # print('STD NMI:', np.std(test_nmi_list))

    # save metrics to csv
    # df = pd.DataFrame(data=cfg.dict_metrics)
    # df.to_csv(os.path.join(cfg.tables, cfg.exp, 'cae_train.csv'), index=False)

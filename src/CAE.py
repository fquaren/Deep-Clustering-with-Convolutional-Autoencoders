from predict import pred_cae
import random
import os
from build_features import get_filenames_list, create_tensors
import config as cfg
import pandas as pd


def pretrainCAE(
        model, x_train, x_val, batch_size, pretrain_epochs, my_callbacks,
        cae_models, optim):
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
    autoencoder.save_weights(os.path.join(cae_models, 'cae_weights'))
    encoder.save_weights(os.path.join(cae_models, 'ce_weights'))
    # save plot metrics
    cfg.d_cae['train_loss'] = autoencoder.history.history['loss']
    cfg.d_cae['val_loss'] = autoencoder.history.history['val_loss']


if __name__ == "__main__":
    # get datasets
    directories, file_list = get_filenames_list(cfg.processed_data)
    x_train, y_train, x_val, y_val, x_test, y_test = create_tensors(
        file_list, directories)

    # pretrain CAE with CAE small
    pretrainCAE(
        model=cfg.cae_small,
        x_train=x_train,
        x_val=x_val,
        batch_size=cfg.cae_batch_size,
        pretrain_epochs=cfg.pretrain_epochs,
        my_callbacks=cfg.my_callbacks,
        cae_models=cfg.cae_models,
        optim=cfg.optim
    )

    pred_cae(
        net=cfg.cae_small,
        weights=os.path.join(cfg.models, cfg.exp, 'cae', 'cae_weights'),
        directory=cfg.test_data,
        scans=cfg.scans,
        figures=cfg.figures,
        exp=cfg.exp,
        n=random.randint(0, 20),
        n_train='primo_train'
    )

    # save metrics to csv
    # df = pd.DataFrame(data=cfg.d_cae)
    # df.to_csv(
    #     os.path.join(cfg.tables, 'cae_train_metrics.csv'), index=False)

    # Transfer learning
    autoencoder, _ = cfg.cae_small
    autoencoder.load_weights(cfg.ce_weights)
    conv1 = autoencoder.get_layer(name='conv1').get_weights()
    conv2 = autoencoder.get_layer(name='conv2').get_weights()
    flatten_1 = autoencoder.get_layer(name='flatten_1').get_weights()

    autoencoder, _ = cfg.cae
    autoencoder.get_layer(name='conv1').set_weights(conv1)
    autoencoder.get_layer(name='conv1').trainable = False
    autoencoder.get_layer(name='conv2').set_weights(conv2)
    autoencoder.get_layer(name='conv2').trainable = False
    autoencoder.get_layer(name='flatten_1').set_weights(flatten_1)
    autoencoder.get_layer(name='flatten_1').trainable = False

    pretrainCAE(
        model=cfg.cae,
        x_train=x_train,
        x_val=x_val,
        batch_size=cfg.cae_batch_size,
        pretrain_epochs=cfg.pretrain_epochs,
        my_callbacks=cfg.my_callbacks,
        cae_models=cfg.cae_models,
        optim=cfg.optim
    )

    pred_cae(
        net=cfg.cae,
        weights=os.path.join(cfg.models, cfg.exp, 'cae', 'cae_weights'),
        directory=cfg.test_data,
        scans=cfg.scans,
        figures=cfg.figures,
        exp=cfg.exp,
        n=random.randint(0, 20),
        n_train='secondo_train'
    )

    # save metrics to csv
    # df = pd.DataFrame(data=cfg.d_cae)
    # df.to_csv(
    #     os.path.join(cfg.tables, 'cae_train_metrics.csv'), index=False)

    autoencoder.get_layer(name='conv1').trainable = True
    autoencoder.get_layer(name='conv2').trainable = True
    autoencoder.get_layer(name='flatten_1').trainable = True

    pretrainCAE(
        model=cfg.cae,
        x_train=x_train,
        x_val=x_val,
        batch_size=cfg.cae_batch_size,
        pretrain_epochs=cfg.pretrain_epochs,
        my_callbacks=cfg.my_callbacks,
        cae_models=cfg.cae_models,
        optim=cfg.optim
    )

    # save metrics to csv
    # df = pd.DataFrame(data=cfg.d_cae)
    # df.to_csv(
    #     os.path.join(cfg.tables, 'cae_train_metrics.csv'), index=False)

    # predict for all categories on test dataset
    pred_cae(
        net=cfg.cae,
        weights=os.path.join(cfg.models, cfg.exp, 'cae', 'cae_weights'),
        directory=cfg.test_data,
        scans=cfg.scans,
        figures=cfg.figures,
        exp=cfg.exp,
        n=random.randint(0, 20),
        n_train='terzo_train'
    )

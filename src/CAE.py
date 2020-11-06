from predict import pred_cae
import random
import os
from build_features import get_filenames_list, create_tensors
import config as cfg
import pandas as pd


def pretrainCAE(
        model, x_train, x_val, batch_size, pretrain_epochs, my_callbacks,
        cae_models):
    autoencoder, encoder = model
    encoder.summary()
    autoencoder.summary()
    autoencoder.compile(optimizer=cfg.optim, loss='mse')
    autoencoder.fit(
        x_train,
        x_train,
        batch_size=batch_size,
        epochs=pretrain_epochs,
        validation_data=(x_val, x_val),
        callbacks=my_callbacks,
    )
    autoencoder.save_weights(
        cae_models, 'cae_weights')
    # save plot metrics
    cfg.d_cae['train_loss'] = autoencoder.history.history['loss']
    cfg.d_cae['val_loss'] = autoencoder.history.history['val_loss']


if __name__ == "__main__":
    # get datasets
    directories, file_list = get_filenames_list(cfg.processed_data)
    x_train, y_train, x_val, y_val, x_test, y_test = create_tensors(
        file_list, directories)

    # pretrain CAE
    #if not os.path.join(cfg.cae_models, 'cae_weights'):
    pretrainCAE(
        model=cfg.cae,
        x_train=x_train,
        x_val=x_val,
        batch_size=cfg.cae_batch_size,
        pretrain_epochs=cfg.pretrain_epochs,
        my_callbacks=cfg.my_callbacks,
        cae_models=cfg.cae_models
    )
    # save metrics to csv
    df = pd.DataFrame(data=cfg.d_cae)
    df.to_csv(
        os.path.join(cfg.tables, 'cae_train_metrics.csv'), index=False)

    # predict for all categories on test dataset
    n = random.randint(0, 100)
    pred_cae(
        net=cfg.cae,
        weights=os.path.join(cfg.models, cfg.exp, 'cae', 'cae_weights'),
        directory=cfg.test_data,
        scans=cfg.scans,
        figures=os.path.join(cfg.figures, cfg.exp, 'cae'),
        exp=cfg.exp,
        n=n
    )

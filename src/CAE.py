from predict import pred
import random
import os
from features.build_features import get_filenames_list, create_tensors
import config as cfg


def pretrainCAE(x_train, x_val, batch_size, pretrain_epochs, my_callbacks):
    autoencoder, encoder = cfg.model
    encoder.summary()
    autoencoder.summary()
    autoencoder.compile(optimizer=cfg.optim, loss=cfg.cae_loss)
    autoencoder.fit(
        x_train,
        x_train,
        batch_size=batch_size,
        epochs=pretrain_epochs,
        validation_data=(x_val, x_val),
        callbacks=my_callbacks,
    )


if __name__ == "__main__":
    # get datasets
    directories, file_list = get_filenames_list(cfg.processed_data)
    x_train, y_train, x_val, y_val, x_test, y_test = create_tensors(
        file_list, directories)
    # pretrain CAE
    pretrainCAE(
        x_train,
        x_val,
        cfg.cae_batch_size,
        cfg.pretrain_epochs,
        cfg.my_callbacks
    )
    # predict for all categories on test dataset
    n = random.randint(0, 100)
    pred(
        cfg.net,
        os.path.join(cfg.models, cfg.exp),
        cfg.test_data,
        cfg.scans,
        cfg.figures,
        cfg.exp,
        n
    )
    pass

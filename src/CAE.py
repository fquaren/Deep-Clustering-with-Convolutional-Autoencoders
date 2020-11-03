from predict import pred
import random
import os
from build_features import get_filenames_list, create_tensors
import config as cfg
from matplotlib import pyplot as plt


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
    # save plot metrics
    plt.figure(figsize=(10, 7))
    plt.plot(autoencoder.history.history['loss'])
    plt.plot(autoencoder.history.history['val_loss'])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train_loss', 'val_loss'])
    plt.savefig(os.path.join(cfg.figures, cfg.exp))


if __name__ == "__main__":
    # get datasets
    directories, file_list = get_filenames_list(cfg.processed_data)
    x_train, y_train, x_val, y_val, x_test, y_test = create_tensors(
        file_list, directories)

    # pretrain CAE
    if not os.path.join(cfg.models, cfg.exp, 'cae_weights'):
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
        cfg.model,
        os.path.join(cfg.models, cfg.exp, 'cae_weights'),
        cfg.test_data,
        cfg.scans,
        cfg.figures,
        cfg.exp,
        n
    )

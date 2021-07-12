from predict import pred_ae
import os
import config as cfg
import pandas as pd
from build_and_save_features import load_dataset
import nets
import visualization as viz
import math
from generators import generators
import predict


def pretrain(
    autoencoder,
    x_train,
    x_val,
    batch_size=cfg.ae_batch_size,
    pretrain_epochs=cfg.pretrain_epochs,
    my_callbacks=cfg.my_callbacks,
    optim='adam',
):

    train_generator, val_generator = generators(
        x_train,
        x_val,
        cfg.ae_batch_size
    )

    print('pretraining...')
    autoencoder.summary()
    autoencoder.compile(optimizer=optim, loss='mse')

    # train model
    autoencoder.fit(
        train_generator,
        steps_per_epoch=math.ceil(x_train.shape[0] / batch_size),
        epochs=pretrain_epochs,
        validation_data=val_generator,
        validation_steps=math.ceil(x_val.shape[0] / batch_size),
        callbacks=my_callbacks
    )

    # save plot metrics
    cfg.dict_metrics['pretrain_train_loss'] = autoencoder.history.history['loss']
    cfg.dict_metrics['pretrain_val_loss'] = autoencoder.history.history['val_loss']

    df = pd.DataFrame(data=cfg.dict_metrics)
    df.to_csv(
        os.path.join(
            cfg.tables,
            cfg.exp,
            'ae_train.csv'
        ),
        index=False
    )
    print('weigths and metrics saved.')

    pred_ae(
        net=autoencoder,
        weights=cfg.ae_weights,
        directory=cfg.train_directory,
    )


if __name__ == "__main__":
    print('EXPERIMENT:', cfg.exp)
    # get datasets
    x_train, y_train = load_dataset('x_train.npy', 'y_train.npy')
    x_val, y_val = load_dataset('x_val.npy', 'y_val.npy')
    x_test, y_test = load_dataset('x_test.npy', 'y_test.npy')

    os.makedirs(os.path.join(cfg.experiments, cfg.exp), exist_ok=True)
    os.makedirs(os.path.join(cfg.tables, cfg.exp), exist_ok=True)
    os.makedirs(os.path.join(cfg.figures, cfg.exp, 'ae'), exist_ok=True)
    os.makedirs(os.path.join(cfg.models, cfg.exp, 'ae'), exist_ok=True)

    autoencoder, _ = nets.autoencoder()
    x_train = x_train.reshape(x_train.shape[0], 128, 128, 1)
    x_val = x_val.reshape(x_val.shape[0], 128, 128, 1)
    x_test = x_test.reshape(x_test.shape[0], 128, 128, 1)

    # pretrain
    pretrain(
        autoencoder=autoencoder,
        x_train=x_train,
        x_val=x_val,
    )

    # predict accuracy
    predict.init_kmeans(
        x=x_train,
        x_val=x_test,
        y=y_train,
        y_val=y_test,
        random_state=None,
        weights=cfg.ae_weights
    )

    # visualization
    viz.plot_pretrain_metrics(
        file=os.path.join(cfg.tables, cfg.exp, 'ae_train.csv'),
        save_dir=os.path.join(cfg.figures, cfg.exp, 'ae'),
    )
    viz.plot_ae_tsne(
        nets.encoder(),
        cfg.ae_weights,
        os.path.join(cfg.figures, cfg.exp, 'ae'),
        x_train,
        x_test
    )
    viz.plot_ae_umap(
        nets.encoder(),
        cfg.ae_weights,
        os.path.join(cfg.figures, cfg.exp, 'ae'),
        x_train,
        x_test
    )
    viz.feature_map(scan=cfg.scans[0], exp=cfg.exp, layer=1, depth=8, weights=cfg.ae_weights)
    viz.feature_map(scan=cfg.scans[0], exp=cfg.exp, layer=2, depth=16, weights=cfg.ae_weights)
    viz.feature_map(scan=cfg.scans[0], exp=cfg.exp, layer=3, depth=32, weights=cfg.ae_weights)
    viz.feature_map(scan=cfg.scans[0], exp=cfg.exp, layer=4, depth=16, weights=cfg.ae_weights)

    print('plots pretrain done.')

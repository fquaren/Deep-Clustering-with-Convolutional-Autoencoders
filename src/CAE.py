from predict import pred_ae
import os
import config as cfg
import pandas as pd
from build_and_save_features import load_dataset
import predict
import visualization as viz
import nets


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

    # save metrics to csv
    df = pd.DataFrame(data=cfg.dict_metrics)
    df.to_csv(os.path.join(cfg.tables, cfg.exp, 'cae_train.csv'), index=False)


if __name__ == "__main__":
    # get datasets
    x_train, y_train = load_dataset('x_train.npy', 'y_train.npy')
    x_val, y_val = load_dataset('x_val.npy', 'y_val.npy')
    x_test, y_test = load_dataset('x_test.npy', 'y_test.npy')

    os.makedirs(os.path.join(cfg.experiments, cfg.exp), exist_ok=True)
    os.makedirs(os.path.join(cfg.tables, cfg.exp), exist_ok=True)
    os.makedirs(os.path.join(cfg.figures, cfg.exp, 'ae'), exist_ok=True)
    os.makedirs(os.path.join(cfg.models, cfg.exp, 'ae'), exist_ok=True)

    x_train = x_train.reshape(x_train.shape[0], 128, 128, 1)
    x_val = x_val.reshape(x_val.shape[0], 128, 128, 1)
    x_test = x_test.reshape(x_test.shape[0], 128, 128, 1)
    
    autoencoder, _ = cfg.cae
    
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

    _, _, _, test_acc, _ = predict.init_kmeans(
        x=x_train,
        x_val=x_test,
        y=y_train,
        y_val=y_test,
        random_state=None,
        weights=cfg.cae_weights,
    )

    pred_ae(
        net=autoencoder,
        weights=cfg.cae_weights,
        directory=cfg.train_directory,
    )
    viz.plot_pretrain_metrics(
        file=os.path.join(cfg.tables, cfg.exp, 'cae_train.csv'),
        save_dir=os.path.join(cfg.figures, cfg.exp, 'ae'),
    )
    viz.plot_ae_tsne(
        nets.encoder(),
        cfg.cae_weights,
        os.path.join(cfg.figures, cfg.exp, 'ae'),
        x_train,
        x_test
    )
    viz.plot_ae_umap(
        nets.encoder(),
        cfg.cae_weights,
        os.path.join(cfg.figures, cfg.exp, 'ae'),
        x_train,
        x_test
    )

    viz.feature_map(
        scan=cfg.scans[0],
        exp=cfg.exp,
        layer=1,
        depth=16,
        weights=cfg.cae_weights
    )
    viz.feature_map(
        scan=cfg.scans[0],
        exp=cfg.exp,
        layer=2,
        depth=32,
        weights=cfg.cae_weights
    )
    viz.feature_map(
        scan=cfg.scans[0],
        exp=cfg.exp,
        layer=3,
        depth=32,
        weights=cfg.cae_weights
    )

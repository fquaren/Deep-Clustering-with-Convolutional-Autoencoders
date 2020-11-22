from predict import pred_cae
import random
import os
import config as cfg
import pandas as pd
from keras.models import Model
from sklearn.cluster import KMeans
from nets import ClusteringLayer
from metrics import nmi, ari, acc
from build_and_save_features import load_dataset


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


def init_kmeans(cae, n_clusters, ce_weights, n_init_kmeans, x, y, gamma):

    autoencoder, encoder = cae

    # init DCEC
    clustering_layer = ClusteringLayer(
        n_clusters, name='clustering')(encoder.output)
    model = Model(
        inputs=encoder.input, outputs=[clustering_layer, autoencoder.output])
    model.compile(
        loss=['kld', 'mse'], loss_weights=[gamma, 1], optimizer=cfg.dcec_optim)
    model.summary()

    # Initialize model using k-means centers
    print('k-means...')
    encoder.load_weights(cfg.ce_weights)
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init_kmeans)
    input_cluster = encoder.predict(x)
    y_pred = kmeans.fit_predict(input_cluster)
    y_pred_last = y_pred.copy()
    centers = kmeans.cluster_centers_
    model.get_layer(name='clustering').set_weights([centers])
    print('metrics before training.')
    print(
        'acc = {}; nmi = {}; ari = {}'.format(
            acc(y, y_pred),
            nmi(y, y_pred),
            ari(y, y_pred)
        )
    )

    return model, y_pred_last


if __name__ == "__main__":
    # get datasets
    x_train, y_train = load_dataset('x_train.npy', 'y_train.npy')
    x_val, y_val = load_dataset('x_val.npy', 'y_val.npy')
    x_test, y_test = load_dataset('x_test.npy', 'y_test.npy')

    os.makedirs(os.path.join(cfg.experiments, cfg.exp), exist_ok=True)
    os.makedirs(os.path.join(cfg.tables, cfg.exp), exist_ok=True)
    os.makedirs(os.path.join(cfg.figures, cfg.exp, 'cae'), exist_ok=True)
    os.makedirs(os.path.join(cfg.models, cfg.exp, 'cae'), exist_ok=True)
    
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

    pred_cae(
        net=cfg.cae,
        weights=os.path.join(cfg.models, cfg.exp, 'cae', 'cae_weights'),
        directory=cfg.train_directory,
        scans=cfg.scans,
        figures=cfg.figures,
        exp=cfg.exp,
        n=random.randint(0, 20),
        n_train='first_pretrain_'
    )

    # save metrics to csv
    df = pd.DataFrame(data=cfg.d_cae)
    df.to_csv(
        os.path.join(cfg.tables, cfg.exp, 'cae_train.csv'), index=False)

    _, _ = init_kmeans(
        cae=cfg.cae,
        n_clusters=cfg.n_clusters,
        ce_weights=cfg.ce_weights,
        n_init_kmeans=cfg.n_init_kmeans,
        x=x_train,
        y=y_train,
        gamma=cfg.gamma
    )

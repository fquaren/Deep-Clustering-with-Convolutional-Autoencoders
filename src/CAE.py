from predict import pred_cae
import random
import os
from build_features import get_filenames_list, create_tensors
import config as cfg
import pandas as pd
from keras.models import Model
from sklearn.cluster import KMeans
from nets import ClusteringLayer
from metrics import nmi, ari, acc


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
        loss=['kld', 'mse'], loss_weights=[gamma, 1], optimizer='adam')
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
    directories, file_list = get_filenames_list(cfg.processed_data)
    x_train, y_train, x_val, y_val, x_test, y_test = create_tensors(
        file_list, directories)

    # pretrain CAE with CAE small
    if not os.path.join(cfg.models, cfg.exp, 'cae', 'cae_weights'):
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
            n_train='first_train_'
        )

        # save metrics to csv
        df = pd.DataFrame(data=cfg.d_cae)
        df.to_csv(
            os.path.join(cfg.tables, 'cae_first_train_metrics.csv'), index=False)

    _, _ = init_kmeans(
        cae=cfg.cae,
        n_clusters=cfg.n_clusters,
        ce_weights=cfg.ce_weights,
        n_init_kmeans=cfg.n_init_kmeans,
        x=x_test,
        y=y_test,
        gamma=cfg.gamma
    )

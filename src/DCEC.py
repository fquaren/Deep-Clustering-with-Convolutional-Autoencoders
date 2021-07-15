from tqdm import tqdm
import os
import numpy as np
import random
import pandas as pd
import config as cfg
from metrics import target_distribution, nmi, ari, acc
from predict import pred_dcec
from build_and_save_features import load_dataset
import predict
from keras.models import Model
from nets import ClusteringLayer
import visualization as viz


def train_val_DCEC(
    maxiter,
    update_interval,
    save_interval,
    x_train,
    y_train,
    model,
    encoder,
    autoencoder,
    tol,
    index,
    dcec_bs,
    dictionary,
    path_models_dcec,
    tables,
    exp,
    y_pred_last,
):

    # Init loss
    train_loss = [0, 0, 0]
    index = 0

    # Train and val
    for ite in tqdm(range(int(maxiter))):

        if ite % update_interval == 0:
            q, _ = model.predict(x_train)
            p = target_distribution(q)

            # Evaluate the clustering performance
            y_train_pred = q.argmax(1)
            if y_train is not None:
                train_acc = np.round(acc(y_train, y_train_pred), 5)
                train_nmi = np.round(nmi(y_train, y_train_pred), 5)
                train_ari = np.round(ari(y_train, y_train_pred), 5)
                train_loss = np.round(train_loss, 5)
                print('\nIter {}: train acc={}, train nmi={}, train ari={}, train loss={}'.format(
                    ite, train_acc, train_nmi, train_ari, train_loss))

            delta_label = np.sum(y_train_pred != y_pred_last).astype(
                np.float32) / y_train_pred.shape[0]
            y_pred_last = np.copy(y_train_pred)
            if ite > 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print('Reached tolerance threshold. Stopping training.')
                # Save the trained model
                model.save_weights(
                    os.path.join(path_models_dcec, 'dcec_model_final.h5'))
                encoder.save_weights(
                    os.path.join(path_models_dcec, 'dcec_encoder_final.h5'))
                break

        if (index + 1) * dcec_bs > x_train.shape[0]:
            train_loss = model.train_on_batch(
                x=x_train[index * dcec_bs::],
                y=[p[index * dcec_bs::], x_train[index * dcec_bs::]]
            )
            index = 0
        else:
            train_loss = model.train_on_batch(
                x=x_train[index * dcec_bs:(index + 1) * dcec_bs],
                y=[
                    p[index * dcec_bs:(index + 1) * dcec_bs], 
                    x_train[index * dcec_bs:(index + 1) * dcec_bs]
                    ]
                )
            index += 1

        # Save metrics to dict for csv
        dictionary['iteration'].append(ite)
        dictionary['train_loss'].append(train_loss[0])
        dictionary['clustering_loss'].append(train_loss[1])
        dictionary['reconstruction_loss'].append(train_loss[2])

        # Save model checkpoint
        if ite % save_interval == 0:
            os.makedirs(os.path.join(exp, path_models_dcec), exist_ok=True)
            model.save_weights(
                os.path.join(
                    exp, path_models_dcec, 'dcec_model_'+str(ite)+'.h5'))
        ite += 1

    # Save metrics to csv
    df = pd.DataFrame(data=dictionary)
    try:
        os.remove(os.path.join(tables, exp, 'dcec_train_metrics.csv'))
    except:
        pass
    df.to_csv(os.path.join(
        tables, exp, 'dcec_train_metrics.csv'), index=False)


def main():
    try:
        os.makedirs(os.path.join(cfg.experiments, cfg.exp))
        os.makedirs(os.path.join(cfg.tables, cfg.exp))
        os.makedirs(os.path.join(cfg.figures, cfg.exp, 'dcec'))
        os.makedirs(os.path.join(cfg.models, cfg.exp, 'dcec'))
    except:
        print('WARNING: Experiment directories already exists.')

    # Get datasets
    x_train, y_train = load_dataset('x_train.npy', 'y_train.npy')
    x_val, y_val = load_dataset('x_val.npy', 'y_val.npy')
    x_test, y_test = load_dataset('x_test.npy', 'y_test.npy')

    x_train = x_train.reshape(x_train.shape[0], 128, 128, 1)
    x_val = x_val.reshape(x_val.shape[0], 128, 128, 1)
    x_test = x_test.reshape(x_test.shape[0], 128, 128, 1)

    autoencoder, encoder = cfg.cae

    autoencoder.load_weights(cfg.cae_weights)

    clustering_layer = ClusteringLayer(
        n_clusters=3, name='clustering')(encoder.output)
    model = Model(
        inputs=encoder.input,
        outputs=[clustering_layer, autoencoder.output]
    )
    model.compile(
        loss=['kld', 'mse'],
        loss_weights=[cfg.gamma, 1],
        optimizer=cfg.dcec_optim
    )
    model.summary()

    y_pred_last, _, centers, pre_test_acc, pre_test_nmi = predict.init_kmeans(
        x=x_train,
        x_val=x_test,
        y=y_train,
        y_val=y_test,
        random_state=None,
        weights=cfg.cae_weights,
    )

    model.get_layer(name='clustering').set_weights([centers])

    train_val_DCEC(
        exp=cfg.exp,
        maxiter=cfg.maxiter,
        update_interval=cfg.update_interval,
        save_interval=cfg.save_interval,
        x_train=x_train,
        y_train=y_train,
        model=model,
        encoder=encoder,
        autoencoder=autoencoder,
        tol=cfg.tol,
        index=cfg.index,
        dcec_bs=cfg.dcec_bs,
        dictionary=cfg.d,
        path_models_dcec=os.path.join(cfg.models, cfg.exp, 'dcec'),
        tables=cfg.tables,
        y_pred_last=y_pred_last
    )

    y_test_pred, _, _, test_acc, test_nmi = predict.init_kmeans_dcec(
        model=model,
        x=x_train,
        x_val=x_test,
        y=y_train,
        y_val=y_test,
        random_state=None,
        weights=os.path.join(
            cfg.models, cfg.exp, 'dcec', 'dcec_model_final.h5'),
    )

    pred_dcec(
        model=model,
        weights=os.path.join(cfg.models, cfg.exp, 'dcec',
                             'dcec_model_final.h5'),
        directory=cfg.test_directory,
        scans=cfg.scans,
        figures=cfg.figures,
        exp=cfg.exp,
        n=random.randint(0, 20),
    )

    viz.plot_ae_tsne(
        encoder,
        os.path.join(cfg.models, cfg.exp, 'dcec', 'dcec_encoder_final.h5'),
        os.path.join(cfg.figures, cfg.exp),
        x_train,
        x_test,
    )
    # viz.plot_confusion_matrix(y_test, y_test_pred)
    viz.plot_ae_umap(
        encoder,
        os.path.join(cfg.models, cfg.exp, 'dcec', 'dcec_encoder_final.h5'),
        os.path.join(cfg.figures, cfg.exp),
        x_train,
        x_test,
    )

    print('done.')


if __name__ == "__main__":
    main()

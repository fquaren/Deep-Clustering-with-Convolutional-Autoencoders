from tqdm import tqdm
import os
from keras.models import Model
from sklearn.cluster import KMeans
import numpy as np
import random
import pandas as pd
from nets import CAE_Conv2DTranspose, ClusteringLayer
import config as cfg
from metrics import target_distribution, nmi, ari, acc
from build_features import get_filenames_list, create_tensors

# TODO as for CAE main wih get dataset, train DCEC and predict

cae, encoder = CAE_Conv2DTranspose()

# DCEC
clustering_layer = ClusteringLayer(
    cfg.n_clusters, name='clustering')(encoder.output[1])
model = Model(
    inputs=encoder.input, outputs=[clustering_layer, cae.output])
model.compile(loss=['kld', 'mse'], loss_weights=[0.1, 1], optimizer='adam')

# Get datasets
directories, file_list = get_filenames_list(cfg.processed_data)
x_train, y_train, x_val, y_val, x_test, y_test = create_tensors(
    file_list, directories)

# Compile encoder for kmeans
cae.load_weights(cfg.cae_weights)
encoder.compile(loss='kld', optimizer='adam')

# Initialize model using k-means centers
print('initializing model using k-means centers...')
kmeans = KMeans(n_clusters=cfg.n_clusters, n_init=50)
features = encoder.predict(x_train)[1]
y_pred = kmeans.fit_predict(features)
y_pred_last = y_pred.copy()
centers = kmeans.cluster_centers_
model.get_layer(name='clustering').set_weights([centers])

print('metrics before training.')
print(
    'acc = {}; nmi = {}; ari = {}'.format(
        acc(y_train, y_pred),
        nmi(y_train, y_pred),
        ari(y_train, y_pred)
    )
)

# Init
train_loss = [0, 0, 0]
val_loss = [0, 0, 0]

# Train and val
for ite in tqdm(range(int(cfg.maxiter))):
    if ite % cfg.update_interval == 0:
        q, _ = model.predict(x_train, verbose=0)
        p = target_distribution(q)
        val_q, _ = model.predict(x_val, verbose=0)
        val_p = target_distribution(val_q)

        # Evaluate the clustering performance
        y_train_pred = q.argmax(1)
        if y_train is not None:
            train_acc = np.round(acc(y_train, y_train_pred), 5)
            train_nmi = np.round(nmi(y_train, y_train_pred), 5)
            train_ari = np.round(ari(y_train, y_train_pred), 5)
            train_loss = np.round(train_loss, 5)
            print(
                'Iter', ite, ': Acc tr', train_acc, ', nmi tr', train_nmi,
                ', ari tr', train_ari, '; loss tr=', train_loss
            )

        y_val_pred = val_q.argmax(1)
        if y_val is not None:
            val_acc = np.round(acc(y_val, y_val_pred), 5)
            val_nmi = np.round(nmi(y_val, y_val_pred), 5)
            val_ari = np.round(ari(y_val, y_val_pred), 5)
            val_loss = np.round(val_loss, 5)
            print(
                'Iter', ite, ': Acc val', val_acc, ', nmi val', val_nmi,
                ', ari val', val_ari, ', loss val=', val_loss)

        # Check stop criterion on train -> TODO on validation?
        delta_label = np.sum(y_train_pred != y_pred_last).astype(
            np.float32) / y_train_pred.shape[0]
        y_pred_last = np.copy(y_train_pred)
        if ite > 0 and delta_label < cfg.tol:
            print('delta_label ', delta_label, '< tol ', cfg.tol)
            print('Reached tolerance threshold. Stopping training.')
            break

    # Train on batch
    if (cfg.index + 1) * cfg.dcec_bs > x_train.shape[0]:
        train_loss = model.train_on_batch(
            x=x_train[cfg.index * cfg.dcec_bs::],
            y=[
                p[cfg.index * cfg.dcec_bs::],
                x_train[cfg.index * cfg.dcec_bs::]
            ]
        )
        cfg.index = 0
    else:
        train_loss = model.train_on_batch(
            x=x_train[
                cfg.index * cfg.dcec_bs:(cfg.index + 1) * cfg.dcec_bs
                ],
            y=[
                p[cfg.index * cfg.dcec_bs:(cfg.index + 1) * cfg.dcec_bs],
                x_train[cfg.index * cfg.dcec_bs:(cfg.index + 1) * cfg.dcec_bs]
            ]
        )
        cfg.index += 1

    # Validation on batch
    x_val_batch = np.array(random.sample(list(x_val), cfg.dcec_bs))
    val_p_batch = np.array(random.sample(list(val_p), cfg.dcec_bs))
    val_loss = model.test_on_batch(
        x=x_val_batch,
        y=[val_p_batch, x_val_batch]
    )

    # Save metrics to dict for csv
    cfg.d['iteration'].append(ite)
    cfg.d['train_loss'].append(train_loss)
    cfg.d['val_loss'].append(val_loss)
    cfg.d['train_acc'].append(train_acc)
    cfg.d['val_acc'].append(val_acc)
    cfg.d['train_nmi'].append(train_nmi)
    cfg.d['val_nmi'].append(val_nmi)
    cfg.d['train_ari'].append(train_ari)
    cfg.d['val_ari'].append(val_ari)

    # TODO Convert to save the best looking at val_loss?
    # Save model checkpoint
    if ite % cfg.save_interval == 0:
        # print('saving model to:', os.path.join(
        #     cfg.models, cfg.exp, 'dcec_model_' + str(ite) + '.h5'))
        model.save_weights(
            os.path.join(
                cfg.models, cfg.exp, 'dcec', 'dcec_model_' + str(ite) + '.h5'))
    ite += 1

# Save metrics to csv
df = pd.DataFrame(data=cfg.d)
df.to_csv(os.path.join(cfg.tables, 'dcec_train_metrics.csv'), index=False)

# Save the trained model
print('saving model to:', os.path.join(
    cfg.models, cfg.exp, 'dcec_model_final.h5'))
model.save_weights(os.path.join(cfg.models, cfg.exp, 'dcec_model_final.h5'))

print('done.')

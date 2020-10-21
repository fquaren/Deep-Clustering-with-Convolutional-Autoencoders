from tqdm import tqdm
import os
from keras.models import Model
from clustering_layer import ClusteringLayer
from sklearn.cluster import KMeans
import numpy as np
import sys
sys.path.append('../') # TODO do __init__ and fix this
from nets import CAE_Conv2DTranspose
import config as cfg
from metrics import target_distribution, nmi, ari, acc
from features.build_features import get_filenames_list, create_tensors
from matplotlib import pyplot as plt


autoencoder, encoder = CAE_Conv2DTranspose()

clustering_layer = ClusteringLayer(cfg.n_clusters, name='clustering')(encoder.output[1])
model = Model(inputs=encoder.input, outputs=[clustering_layer, autoencoder.output])
# encoder.compile(loss='kld', optimizer='adam')
# model.compile(loss=['kld', 'mse'], loss_weights=[0.1, 1], optimizer='adam')
autoencoder.load_weights(cfg.cae_weights) 

# get test dataset
directories, file_list = get_filenames_list(cfg.processed_data)
x_train, y_train, x_val, y_val, x_test, y_test = create_tensors(file_list, directories)

# Initialize cluster centers using k-means
print('initializing cluster centers using k-means...')
kmeans = KMeans(n_clusters=cfg.n_clusters, n_init=100)

# predict with kmeans
print('predicting with k-means...')
features = encoder.predict(x_train)[1]
y_pred = kmeans.fit_predict(features)
y_pred_last = y_pred.copy()
centers = kmeans.cluster_centers_
n_iter = kmeans.n_iter_

model.get_layer(name='clustering').set_weights([centers])

print('metrics before training.')
print(
    'acc = {}; nmi = {}; ari = {}'.format(
        acc(y_train, y_pred),
        nmi(y_train, y_pred),
        ari(y_train, y_pred)
    )
)
print('numero di iterazioni: {}'.format(n_iter))

# save fig scatter plot
plt.figure(figsize=(30, 10))
plt.subplot(1, 3, 1)
plt.scatter(features[:, 0], features[:, 1], c=y_pred, s=20, cmap='brg')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=1)
plt.xlabel('X')
plt.ylabel('Y')
plt.subplot(1, 3, 2)
plt.scatter(features[:, 1], features[:, 2], c=y_pred, s=20, cmap='brg')
plt.scatter(centers[:, 1], centers[:, 2], c='black', s=200, alpha=1)
plt.xlabel('Y')
plt.ylabel('Z')
plt.subplot(1, 3, 3)
plt.scatter(features[:, 0], features[:, 2], c=y_pred, s=20, cmap='brg')
plt.scatter(centers[:, 0], centers[:, 2], c='black', s=200, alpha=1)
plt.xlabel('X')
plt.ylabel('Z')
plt.savefig(os.path.join(cfg.figures, cfg.exp, 'kmeans_init'))
# TODO Which is which?


# init
index = 0
val_index = 0
train_loss = []
val_loss = []

# plots
history_train_losses = [[], [], []]
history_val_losses = [[], [], []]
history_train_acc = []
history_val_acc = []
train_iterazioni = []
val_iterazioni = []

# compile model
model.compile(loss=['kld', 'mse'], loss_weights=[cfg.gamma, 1], optimizer=cfg.optim)
for ite in tqdm(range(int(cfg.maxiter))):
    if ite % cfg.update_interval == 0:
        q, _ = model.predict(x_train, verbose=0)
        # update the auxiliary target distribution p
        p = target_distribution(q)

        # evaluate the clustering performance
        y_train_pred = q.argmax(1)
        if y_train is not None:
            train_acc = np.round(acc(y_train, y_train_pred), 5)
            train_nmi = np.round(nmi(y_train, y_train_pred), 5)
            train_ari = np.round(ari(y_train, y_train_pred), 5)
            train_loss = np.round(train_loss, 5)
            print('Iter', ite, ': Acc tr', train_acc, ', nmi tr',
                  train_nmi, ', ari tr', train_ari, '; loss tr=', train_loss)

        # check stop criterion
        delta_label = np.sum(y_train_pred != y_pred_last).astype(
            np.float32) / y_train_pred.shape[0]
        y_pred_last = np.copy(y_train_pred)
        if ite > 0 and delta_label < cfg.tol:
            print('delta_label ', delta_label, '< tol ', cfg.tol)
            print('Reached tolerance threshold. Stopping training.')
            break

    # train on batch
    if (index + 1) * cfg.dcec_batch_size > x_train.shape[0]:
        train_loss = model.train_on_batch(
            x=x_train[index * cfg.dcec_batch_size::],
            y=[
                p[index * cfg.dcec_batch_size::],
                x_train[index * cfg.dcec_batch_size::]
            ]
        )
        index = 0
    else:
        train_loss = model.train_on_batch(
            x=x_train[index * cfg.dcec_batch_size:(index + 1) * cfg.dcec_batch_size],
            y=[
                p[index * cfg.dcec_batch_size:(index + 1) * cfg.dcec_batch_size],
                x_train[index * cfg.dcec_batch_size:(index + 1) * cfg.dcec_batch_size]
            ]
        )
        index += 1

    history_train_losses[0].append(train_loss[0])
    history_train_losses[1].append(train_loss[1])
    history_train_losses[2].append(train_loss[2])
    history_train_acc.append(train_acc)

    train_iterazioni.append(ite)

    if ite % cfg.save_interval == 0:
        # save DCEC model checkpoints
        # print('saving model to:', os.path.join(
        #     cfg.models, cfg.exp, 'dcec_model_' + str(ite) + '.h5'))
        model.save_weights(
            os.path.join(cfg.models, cfg.exp, 'dcec_model_' + str(ite) + '.h5'))

    ite += 1

# save the trained model
print('saving model to:', os.path.join(cfg.models, cfg.exp, 'dcec_model_final.h5'))
model.save_weights(os.path.join(cfg.models, 'dcec_model_final.h5'))

plt.figure(figsize=(25, 5))
plt.subplot(1, 4, 1)
x1 = train_iterazioni
y1 = history_train_losses[0]
plt.plot(x1, y1)
plt.title('L')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])

plt.subplot(1, 4, 2)
x1 = train_iterazioni
y1 = history_train_losses[1]
plt.plot(x1, y1)
plt.title('Lr: reconstruction loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])

plt.subplot(1, 4, 3)
x1 = train_iterazioni
y1 = history_train_losses[2]
plt.plot(x1, y1)
plt.title('Lc: clustering loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])

plt.subplot(1, 4, 4)
x1 = train_iterazioni
y1 = history_train_acc
plt.plot(x1, y1)
plt.title('Acc')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend('Train')

plt.savefig(os.path.join(cfg.figures, cfg.exp, 'train_loss_and_acc'))

for ite in range(int(cfg.maxiter)):
    if ite % cfg.update_interval == 0:
        val_q, _ = model.predict(x_val, verbose=0)
        # update the auxiliary target distribution p
        val_p = target_distribution(val_q)

        # evaluate the clustering performance
        y_val_pred = val_q.argmax(1)
        if y_val is not None:
            val_acc = np.round(acc(y_val, y_val_pred), 5)
            val_nmi = np.round(nmi(y_val, y_val_pred), 5)
            val_ari = np.round(ari(y_val, y_val_pred), 5)
            val_loss = np.round(val_loss, 5)
            print('Iter', ite, ': Acc val', val_acc, ', nmi val',
                  val_nmi, ', ari val', val_ari, '; loss val=', val_loss)

        # check stop criterion
        delta_label = np.sum(y_train_pred != y_pred_last).astype(
            np.float32) / y_train_pred.shape[0]
        y_pred_last = np.copy(y_train_pred)
        if ite > 0 and delta_label < cfg.tol:
            print('delta_label ', delta_label, '< tol ', cfg.tol)
            print('Reached tolerance threshold. Stopping training.')
            break

    # train on batch
    if (val_index + 1) * cfg.dcec_batch_size > x_val.shape[0]:
        val_loss = model.test_on_batch(
            x=x_val[val_index * cfg.dcec_batch_size::],                   
            y=[val_p[val_index * cfg.dcec_batch_size::], x_val[val_index * cfg.dcec_batch_size::]]
        )
        val_index = 0
    else:
        val_loss = model.test_on_batch(
            x=x_val[val_index * cfg.dcec_batch_size:(val_index + 1) * cfg.dcec_batch_size],
            y=[
                val_p[val_index * cfg.dcec_batch_size:(val_index + 1) * cfg.dcec_batch_size], 
                x_val[val_index * cfg.dcec_batch_size:(val_index + 1) * cfg.dcec_batch_size]
            ]
        )
        val_index += 1 
    
    history_val_losses[0].append(val_loss[0])
    history_val_losses[1].append(val_loss[1])
    history_val_losses[2].append(val_loss[2])
    history_val_acc.append(val_acc)

    val_iterazioni.append(ite)

    ite += 1

plt.figure(figsize=(25, 5))
plt.subplot(1, 4, 1)
x1 = val_iterazioni
y1 = history_val_losses[0]
plt.plot(x1, y1)
plt.title('L')
plt.ylabel('Val Loss')
plt.xlabel('Epoch')
plt.legend(['Validation'])

plt.subplot(1, 4, 2)
x1 = val_iterazioni
y1 = history_val_losses[1]
plt.plot(x1, y1)
plt.title('Lr: reconstruction loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Validation'])

plt.subplot(1, 4, 3)
x1 = val_iterazioni
y1 = history_val_losses[2]
plt.plot(x1, y1)
plt.title('Lc: clustering loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Validation'])

plt.subplot(1, 4, 4)
x1 = val_iterazioni
y1 = history_val_acc
plt.plot(x1, y1)
plt.title('Acc')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend('Validation')

plt.savefig(os.path.join(cfg.figures, cfg.exp, 'val_loss_and_acc'))

model.load_weights(os.path.join(cfg.models, 'dcec_model_final.h5'))

pred_encoder = model.predict(x_test)[0]

print('done.')

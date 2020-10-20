from tqdm import tqdm
import os
from keras.models import Model
from clustering_layer import ClusteringLayer
from sklearn.cluster import KMeans
import numpy as np
import sys
sys.path.append('../')
from nets import CAE_Conv2DTranspose
import config as cfg
import metrics
from features.build_features import get_filenames_list, create_tensors
from matplotlib import pyplot as plt


autoencoder, encoder = CAE_Conv2DTranspose()

clustering_layer = ClusteringLayer(cfg.n_clusters, name='clustering')(encoder.output[1])
model = Model(inputs=encoder.input, outputs=[clustering_layer, autoencoder.output])
#encoder.compile(loss='kld', optimizer='adam')
#model.compile(loss=['kld', 'mse'], loss_weights=[0.1, 1], optimizer='adam')
autoencoder.load_weights(cfg.cae_weights) 

# get test dataset
directories, file_list = get_filenames_list(cfg.processed_data)
x_train, y_train, x_val, y_val, x_test, y_test = create_tensors(file_list, directories)

# Initialize cluster centers using k-means
print('initializing cluster centers using k-means...')
kmeans = KMeans(n_clusters=cfg.n_clusters, n_init=30)

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
        metrics.acc(y_train, y_pred),
        metrics.nmi(y_train, y_pred),
        metrics.ari(y_train, y_pred)
    )
)
print('numero di iterazioni: {}'.format(n_iter))

# save fig scatter plot
plt.figure(figsize=(30, 10))
plt.subplot(1,3,1)
plt.scatter(features[:, 0], features[:, 1], c=y_pred, s=20, cmap='brg')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=1)
plt.xlabel('X')
plt.ylabel('Y')
plt.subplot(1,3,2)
plt.scatter(features[:, 1], features[:, 2], c=y_pred, s=20, cmap='brg')
plt.scatter(centers[:, 1], centers[:, 2], c='black', s=200, alpha=1)
plt.xlabel('Y')
plt.ylabel('Z')
plt.subplot(1,3,3)
plt.scatter(features[:, 0], features[:, 2], c=y_pred, s=20, cmap='brg')
plt.scatter(centers[:, 0], centers[:, 2], c='black', s=200, alpha=1)
plt.xlabel('X')
plt.ylabel('Z')
plt.savefig(os.path.join(cfg.figures, cfg.exp, 'kmeans_init'))
# TODO Which is which?

# Student's distribution (see paper)
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

# TODO clean from here

# hyperparams
batch_size = 16
maxiter = 10000
update_interval = 5000
save_interval = 1000
tol = 0.01  # tolerance as threshold to early stopping

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
for ite in tqdm(range(int(maxiter))):
    if ite % update_interval == 0:
        q, _ = model.predict(x_train, verbose=0)
        # update the auxiliary target distribution p
        p = target_distribution(q)

        # evaluate the clustering performance
        y_train_pred = q.argmax(1)
        if y_train is not None:
            train_acc = np.round(metrics.acc(y_train, y_train_pred), 5)
            train_nmi = np.round(metrics.nmi(y_train, y_train_pred), 5)
            train_ari = np.round(metrics.ari(y_train, y_train_pred), 5)
            train_loss = np.round(train_loss, 5)
            print('Iter', ite, ': Acc tr', train_acc, ', nmi tr',
                  train_nmi, ', ari tr', train_ari, '; loss tr=', train_loss)

        # check stop criterion
        delta_label = np.sum(y_train_pred != y_pred_last).astype(
            np.float32) / y_train_pred.shape[0]
        y_pred_last = np.copy(y_train_pred)
        if ite > 0 and delta_label < tol:
            print('delta_label ', delta_label, '< tol ', tol)
            print('Reached tolerance threshold. Stopping training.')
            break

    # train on batch
    if (index + 1) * batch_size > x_train.shape[0]:
        train_loss = model.train_on_batch(
            x=x_train[index * batch_size::],
            y=[p[index * batch_size::], x_train[index * batch_size::]]
        )
        index = 0
    else:
        train_loss = model.train_on_batch(
            x=x_train[index * batch_size:(index + 1) * batch_size],
            y=[
                p[index * batch_size:(index + 1) * batch_size], 
                x_train[index * batch_size:(index + 1) * batch_size]
            ]
        )
        index += 1

    history_train_losses[0].append(train_loss[0])
    history_train_losses[1].append(train_loss[1])
    history_train_losses[2].append(train_loss[2])
    history_train_acc.append(train_acc)

    train_iterazioni.append(ite)

    if ite % save_interval == 0:
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

print('done.')
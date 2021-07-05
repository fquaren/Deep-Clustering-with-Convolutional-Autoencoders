import os
from matplotlib import pyplot as plt
import cv2
from glob import glob
import numpy as np
import config as cfg
from build_and_save_features import load_dataset
import nets
from sklearn.cluster import KMeans
import random
from metrics import acc, nmi
from sklearn.manifold import TSNE
from sklearn.decomposition import FastICA, PCA
from sklearn import random_projection, cluster
import umap


def get_list_per_type(path, scan):
    images = glob(os.path.join(path, scan, '*.png'))
    return images


def get_image(names, n):
    image = cv2.imread(names[n])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, dsize=(128, 128))
    return image


def pred_ae(net, weights, directory, scans, figures, exp, n):
    '''
    Predict the output of the net from a test image and save the prediction
    (one for each scan).
    '''
    for scan in scans:
        net.load_weights(weights)
        img_1 = get_image(get_list_per_type(directory, scan), n)
        img_2 = get_image(get_list_per_type(directory, scan), n-1)
        images = [img_1, img_2]
        img = np.stack(images)
        pred_img = net.predict(img)
        plt.figure(figsize=(14, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(img[0])
        plt.subplot(1, 2, 2)
        plt.imshow(pred_img[0])
        os.makedirs(os.path.join(figures, exp, 'ae'), exist_ok=True)
        plt.savefig(os.path.join(figures, exp, 'ae', scan+'_ae_pred.png'))
    print('Prediction on test images done.')


def init_kmeans(x, y, n_clusters=3, n_init_kmeans=100, verbose=True, weights=cfg.ce_weights):
    encoder = nets.encoder()
    encoder.load_weights(weights)   
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init_kmeans)
    embedding = encoder.predict(x)
    y_pred = kmeans.fit_predict(embedding)
    centers = kmeans.cluster_centers_
    if verbose:
        print('metrics:')
        print('acc = {}; nmi = {}'.format(acc(y, y_pred), nmi(y, y_pred)))
    cfg.d_ae['acc'] = acc(y, y_pred)
    cfg.d_ae['nmi'] = nmi(y, y_pred)

    return y_pred, centers


def init_kmeans_on_projection(x, y, n_clusters=3, n_init_kmeans=100, verbose=True, weights=cfg.ce_weights):
    _, encoder = nets.autoencoder()
    encoder.load_weights(weights)
    features, _ = encoder.predict(x)
    transformer = cluster.FeatureAgglomeration(n_clusters=3)
    embedding = transformer.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init_kmeans)
    y_pred = kmeans.fit_predict(embedding)
    centers = kmeans.cluster_centers_
    if verbose:
        print('metrics:')
        print('acc = {}; nmi = {}'.format(acc(y, y_pred), nmi(y, y_pred)))
    cfg.d_ae['acc'] = acc(y, y_pred)
    cfg.d_ae['nmi'] = nmi(y, y_pred)  

    return y_pred, centers


if __name__ == "__main__":
    # get datasets
    x_train, y_train = load_dataset('x_train.npy', 'y_train.npy')
    x_val, y_val = load_dataset('x_val.npy', 'y_val.npy')
    x_test, y_test = load_dataset('x_test.npy', 'y_test.npy')

    autoencoder, encoder = nets.autoencoder()
    x_train = x_train.reshape(x_train.shape[0], 128, 128, 1)
    x_val = x_val.reshape(x_val.shape[0], 128, 128, 1)
    x_test = x_test.reshape(x_test.shape[0], 128, 128, 1)

    #encoder.load_weights(cfg.ce_weights)
    autoencoder.load_weights(cfg.ae_weights)

    pred_ae(
        net=autoencoder,
        weights=os.path.join(cfg.models, cfg.exp, 'ae', 'ae_weights'),
        directory=cfg.test_directory,
        scans=cfg.scans,
        figures=cfg.figures,
        exp=cfg.exp,
        n=random.randint(0, 10)
    )


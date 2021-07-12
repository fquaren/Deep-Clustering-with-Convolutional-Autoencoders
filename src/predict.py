import os
from matplotlib import pyplot as plt
import cv2
from glob import glob
import numpy as np
import config as cfg
import random
import nets
from sklearn.cluster import KMeans
from metrics import acc, nmi


def get_list_per_type(path, scan):
    images = glob(os.path.join(path, scan, '*.png'))
    return images


def get_image(names, n):
    image = cv2.imread(names[n])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, dsize=(128, 128))
    return image


# def pred_cae(net, weights, directory, scans, figures, exp, n):
#     '''
#     Predict the output of the net from a test image and save the prediction
#     (one for each scan).
#     '''
#     for scan in scans:
#         autoencoder, encoder = net
#         autoencoder.load_weights(weights)
#         img = get_image(get_list_per_type(directory, scan), n)
#         img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_LANCZOS4)
#         pred_img = autoencoder.predict(img.reshape((1,) + img.shape + (1,)))
#         pred_img = pred_img.reshape((128, 128))
#         # plot prediction and save image
#         plt.figure(figsize=(14, 7))
#         plt.subplot(1, 2, 1)
#         plt.imshow(img)
#         plt.subplot(1, 2, 2)
#         plt.imshow(pred_img)
#         os.makedirs(os.path.join(figures, exp, 'cae'), exist_ok=True)
#         plt.savefig(os.path.join(figures, exp, 'cae', scan+'_cae_pred.png'))
#     print('Prediction on test images done.')


def pred_ae(net, weights, directory, exp=cfg.exp, n=random.randint(0, 10), scans=cfg.scans, figures=cfg.figures,):
    '''
    Predict the output of the net from a test image and save the prediction
    (one for each scan).
    '''
    net.load_weights(weights)
    for scan in scans:
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
        plt.close()
    print('Prediction on test images done.')
    

def pred_dcec(model, weights, directory, scans, figures, exp, n):
    '''
    Predict the output of the net from a test image and save the prediction
    (one for each scan).
    '''
    for scan in scans:
        model.load_weights(weights)
        img = get_image(get_list_per_type(directory, scan), n)
        img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_LANCZOS4)
        pred_img = model.predict(img.reshape((1,) + img.shape + (1,)))
        pred_img = pred_img[1].reshape((128, 128))
        # plot prediction and save image
        plt.figure(figsize=(14, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.imshow(pred_img)
        os.makedirs(os.path.join(figures, exp, 'dcec'), exist_ok=True)
        plt.savefig(os.path.join(figures, exp, 'dcec', scan))
        plt.close()
    print('Prediction on test images done.')
    

def init_kmeans(x, x_val, y, y_val, random_state, weights, n_clusters=3, verbose=True):
    kmeans = KMeans(n_clusters=n_clusters, n_init=100)
    encoder = nets.encoder()
    encoder.load_weights(weights)
    kmeans = kmeans
    kmeans.random_state = random_state
    embedding = encoder.predict(x)
    y_pred = kmeans.fit_predict(embedding)
    centers = kmeans.cluster_centers_
    val_embedding = encoder.predict(x_val)
    y_val_pred = kmeans.predict(val_embedding)

    if verbose:
        print('metrics:')
        print('TRAIN ACC = {}; TRAIN NMI = {}'.format(
            np.round(acc(y, y_pred), 3),
            np.round(nmi(y, y_pred), 3)
            )
        )
        print('TEST ACC = {}; TEST NMI = {}'.format(
            np.round(acc(y_val, y_val_pred), 3),
            np.round(nmi(y_val, y_val_pred), 3)
            )
        )
        print('RANDOM STATE:', random_state)

    cfg.dict_metrics['train_acc'] = acc(y, y_pred)
    cfg.dict_metrics['train_nmi'] = nmi(y, y_pred)
    cfg.dict_metrics['val_acc'] = acc(y_val, y_val_pred)
    cfg.dict_metrics['val_nmi'] = nmi(y_val, y_val_pred)

    test_acc = acc(y_val, y_val_pred)
    test_nmi = nmi(y_val, y_val_pred)
    
    return y_pred, y_val_pred, centers, test_acc, test_nmi


def init_kmeans_dcec(model, x, x_val, y, y_val, random_state, weights, n_clusters=3, verbose=True):
    kmeans = KMeans(n_clusters=n_clusters, n_init=100)
    model.load_weights(weights)
    kmeans = kmeans
    kmeans.random_state = random_state
    embedding = model.predict(x)[0]
    y_pred = kmeans.fit_predict(embedding)
    centers = kmeans.cluster_centers_
    val_embedding = model.predict(x_val)[0]
    y_val_pred = kmeans.predict(val_embedding)

    if verbose:
        print('metrics:')
        print('TRAIN ACC = {}; TRAIN NMI = {}'.format(
            np.round(acc(y, y_pred), 3),
            np.round(nmi(y, y_pred), 3)
            )
        )
        print('TEST ACC = {}; TEST NMI = {}'.format(
            np.round(acc(y_val, y_val_pred), 3),
            np.round(nmi(y_val, y_val_pred), 3)
            )
        )
        print('RANDOM STATE:', random_state)

    cfg.dict_metrics['train_acc'] = acc(y, y_pred)
    cfg.dict_metrics['train_nmi'] = nmi(y, y_pred)
    cfg.dict_metrics['val_acc'] = acc(y_val, y_val_pred)
    cfg.dict_metrics['val_nmi'] = nmi(y_val, y_val_pred)

    test_acc = acc(y_val, y_val_pred)
    test_nmi = nmi(y_val, y_val_pred)
    
    return y_pred, y_val_pred, centers, test_acc, test_nmi

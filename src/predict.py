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


def get_list_per_type(path, scan):
    images = glob(os.path.join(path, scan, '*.png'))
    return images


def get_image(names, n):
    image = cv2.imread(names[n])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, dsize=(128, 128))
    return image


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
    print('Prediction on test images done.')
    plt.close()


def init_kmeans(x, x_val, y, y_val, random_state, weights, n_clusters=3, verbose=True):
    encoder = nets.encoder()
    encoder.load_weights(weights)
    kmeans = cfg.kmeans
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

    cfg.dict_metrics['train_acc'] = acc(y, y_pred)
    cfg.dict_metrics['train_nmi'] = nmi(y, y_pred)
    cfg.dict_metrics['val_acc'] = acc(y_val, y_val_pred)
    cfg.dict_metrics['val_nmi'] = nmi(y_val, y_val_pred)

    return y_pred, y_val_pred, centers


if __name__ == "__main__":
    # get datasets
    x_train, y_train = load_dataset('x_train.npy', 'y_train.npy')
    x_val, y_val = load_dataset('x_val.npy', 'y_val.npy')
    x_test, y_test = load_dataset('x_test.npy', 'y_test.npy')

    autoencoder, encoder = nets.autoencoder()
    x_train = x_train.reshape(x_train.shape[0], 128, 128, 1)
    x_val = x_val.reshape(x_val.shape[0], 128, 128, 1)
    x_test = x_test.reshape(x_test.shape[0], 128, 128, 1)

    autoencoder.load_weights(cfg.ae_weights)

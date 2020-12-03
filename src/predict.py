import os
from matplotlib import pyplot as plt
import cv2
from glob import glob


def get_list_per_type(path, scan):
    #images = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]
    images = glob(os.path.join(path, '**', '*.png'))
    images = [f for f in images if scan in f]
    return images


def get_image(names, n):
    image = cv2.imread(names[n])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def pred_cae(net, weights, directory, scans, figures, exp, n):
    '''
    Predict the output of the net from a test image and save the prediction
    (one for each scan).
    '''
    for scan in scans:
        autoencoder, encoder = net
        autoencoder.load_weights(weights)
        img = get_image(get_list_per_type(directory, scan), n)
        img = cv2.resize(
            img, dsize=(192, 192), interpolation=cv2.INTER_LANCZOS4)
        pred_img = autoencoder.predict(img.reshape((1,) + img.shape + (1,)))
        pred_img = pred_img.reshape((192, 192))
        # plot prediction and save image
        plt.figure(figsize=(14, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.imshow(pred_img)
        os.makedirs(os.path.join(figures, exp, 'cae'), exist_ok=True)
        plt.savefig(os.path.join(figures, exp, 'cae', scan+'_cae_pred.png'))
    print('Prediction on test images done.')


def pred_dcec(model, weights, directory, scans, figures, exp, n):
    '''
    Predict the output of the net from a test image and save the prediction
    (one for each scan).
    '''
    for scan in scans:
        model.load_weights(weights)
        img = get_image(get_list_per_type(directory, scan), n)
        img = cv2.resize(
            img, dsize=(192, 192), interpolation=cv2.INTER_LANCZOS4)
        pred_img = model.predict(img.reshape((1,) + img.shape + (1,)))[1]
        pred_img = pred_img.reshape((192, 192))
        # plot prediction and save image
        plt.figure(figsize=(14, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.imshow(pred_img)
        os.makedirs(os.path.join(figures, exp, 'dcec'), exist_ok=True)
        plt.savefig(os.path.join(figures, exp, 'dcec', scan))
    print('Prediction on test images done.')

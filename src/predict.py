import os
from matplotlib import pyplot as plt
import cv2
from glob import glob
import numpy as np


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
        #img = np.rollaxis(img, -1)
        # import pdb; pdb.set_trace()
        
        pred_img = net.predict(img)

        # plot prediction and save image
        plt.figure(figsize=(14, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(img[0])
        plt.subplot(1, 2, 2)
        plt.imshow(pred_img[0])
        os.makedirs(os.path.join(figures, exp, 'ae'), exist_ok=True)
        plt.savefig(os.path.join(figures, exp, 'ae', scan+'_ae_pred.png'))
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
    print('Prediction on test images done.')

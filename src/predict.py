import os
from matplotlib import pyplot as plt
import cv2
from metrics import target_distribution, acc


def get_list_per_type(directory, scan):
    names = [f for f in os.listdir(directory) if f[:2] == scan[:2]]
    return names


def get_image(names, directory, n):
    image = cv2.imread(os.path.join(directory, names[n]))
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
        img = get_image(get_list_per_type(directory, scan), directory, n)
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


# TODO

def pred_dcec(model, weights, scans, directory, figures, exp, n):
    for scan in scans:
        model.load_weights(weights)
        img = get_image(get_list_per_type(directory, scan), directory, n)
        img = cv2.resize(
            img, dsize=(192, 192), interpolation=cv2.INTER_LANCZOS4)
        pred_img = model.predict(img.reshape((1,) + img.shape + (1,)))
        pred_img = pred_img.reshape((192, 192))
        # plot prediction and save image
        plt.figure(figsize=(14, 7))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.imshow(pred_img)
        plt.savefig(os.path.join(figures, exp, scan+'_cae_pred.png'))
    print('Prediction on test images done.')


def test_dcec(model, x_test, y_test):
    test_q, _ = model.predict(x_test, verbose=0)
    test_p = target_distribution(test_q)
    test_loss = model.fit(x=x_test, y=[test_p, x_test], verbose=0)
    test_acc = []
    y_test_pred = test_q.argmax(1)
    test_acc = acc(y_test, y_test_pred)
    return test_loss, test_acc

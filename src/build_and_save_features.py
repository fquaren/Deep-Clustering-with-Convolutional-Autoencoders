# Build feature for modelling.

import cv2
import numpy as np
from tqdm import tqdm
import config as cfg
import os
import glob


def read_images(path):
    '''
    Reads the images contained in path, creates two tensors (x, y) with images
    and labels.
    N.B.: the images are reshaped.

    input: list of image paths
    output: (x, y)
        x.shape: (N, 192, 192, 1)
        y.shape: (N, )
    '''
    labels = []
    images = []
    for i in tqdm(path):
        if 'CT' in i:
            labels.append(0)
        if 'MR' in i:
            labels.append(1)
        if 'PET' in i:
            labels.append(2)
        img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, dsize=(128, 128),
                         interpolation=cv2.INTER_LANCZOS4)
        images.append(img)
    y = np.concatenate((labels,))
    x = np.dstack(images)
    x = np.rollaxis(x, -1)
    x = x/255.
    return x, y


def create_tensors(path):
    '''
    uses the read_images function to create the tensors and labels for train,
    validation and test dataset.
    '''
    X = []
    Y = []

    images = glob.glob(path+'/**.png')
    # normalization done at the train directory level
    X, Y = read_images(images)
    X = np.dstack(X)
    X = np.rollaxis(X, -1)
    X -= np.mean(X)  # mean subtraction
    # X /= np.std(X, axis = 0)  # normalization
    X = X.reshape(X.shape + (1,))
    return X, Y


def save_images_to_numpy(x_train, y_train, x_val, y_val, x_test, y_test, path):
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, 'x_train.npy'), x_train)
    np.save(os.path.join(path, 'x_val.npy'), x_val)
    np.save(os.path.join(path, 'x_test.npy'), x_test)
    np.save(os.path.join(path, 'y_train.npy'), y_train)
    np.save(os.path.join(path, 'y_val.npy'), y_val)
    np.save(os.path.join(path, 'y_test.npy'), y_test)
    print('Images saved to {} as numpy arrays.'.format(path))


def load_dataset(x, y):
    x = np.load(os.path.join(cfg.numpy, x))
    y = np.load(os.path.join(cfg.numpy, y))
    return (x, y)


def main():
    # list file in directories
    x_train, y_train = create_tensors(cfg.train_directory)
    x_val, y_val = create_tensors(cfg.val_directory)
    x_test, y_test = create_tensors(cfg.test_directory)
    print('Train: {}\nValidation: {}\nTest: {}'.format(
        x_train.shape, x_val.shape, x_test.shape))
    save_images_to_numpy(
        x_train, y_train, x_val, y_val, x_test, y_test, cfg.numpy)


if __name__ == "__main__":
    main()

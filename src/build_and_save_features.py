# Build feature for modelling.

import random
from os import listdir
import cv2
import numpy as np
from skimage import io
from tqdm import tqdm
import config as cfg
import os

processed_data = cfg.processed_data


def get_filenames_list(processed_data):
    '''
    get list of filenames of images to create the train, val and test datasets
    with.
    '''
    directories = [
        processed_data+'train/',
        processed_data+'val/',
        processed_data+'test/'
    ]

    # train_file_names = [f for f in listdir(
    #     directories[0]) if isfile(join(directories[0], f))]
    # val_file_names = [f for f in listdir(
    #     directories[1]) if isfile(join(directories[1], f))]
    # test_file_names = [f for f in listdir(
    #     directories[2]) if isfile(join(directories[2], f))]

    # _file_names = [train_file_names, val_file_names, test_file_names]

    # numero immagini per categoria
    scans = ['CT', 'MRI', 'PET']
    numbers = []
    minimi = []

    for directory in directories:
        for scan in scans:
            a = len([f for f in listdir(directory) if f[:2] == scan[:2]])
            print('Numbero di immagini', scan, 'in', directory, ':', a)
            numbers.append(a)
        minimi.append(min(numbers))

    # creo la lista di file bilanciata: n.b  la percentuale viene mantenuta
    train_final_file_names = []
    val_final_file_names = []
    test_final_file_names = []
    _final_file_names = [
        train_final_file_names,
        val_final_file_names,
        test_final_file_names
    ]

    lista = []
    for directory, minimo, name in zip(directories, minimi, _final_file_names):
        for scan in scans:
            lista = [f for f in listdir(directory) if f[:2] == scan[:2]]
            lista = lista[:minimo]
            name.extend(lista)  # estendo la lista (don't append)

# -------------------------------

    # How I cicled ^^^
    #    minimo |
    #    name   |
    #    train  | val | test
    # CT    ... | ... | ...
    # MRI   ... | ... | ...
    # PET   ... | ... | ...

# ---------------------------------

    for el in _final_file_names:
        random.shuffle(el)

    print("FINAL DATASETS")
    print("Training", len(train_final_file_names), ", Validation:", len(
        val_final_file_names), ", Test:", len(test_final_file_names))

    return directories, _final_file_names


def read_images(path):
    '''
    Reads the images contained in path, creates two tensors (x, y) with images
    and labels.
    N.B.: the images are reshaped and normalised.

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
        img = cv2.resize(img, dsize=(144, 144), interpolation=cv2.INTER_LANCZOS4)
        images.append(img)
    y = np.concatenate((labels,))
    x = np.dstack(images)
    x = np.rollaxis(x, -1)
    # x = x.reshape(x.shape + (1,))
    # x = x/255.
    import pdb; pdb.set_trace()
    x = x - np.mean(x)  # normalization
    x = x / np.std(x)
    return x, y


def create_tensors(path):
    '''
    uses the read_images function to create the tensors and labels for train,
    validation and test dataset.
    '''
    X = []
    Y = []
    for directory in os.listdir(path):
        images = [os.path.join(path, directory, f) for f in os.listdir(os.path.join(path, directory)) if f.endswith('.png')]
        x, y = read_images(images)  # normalization done at the scan level
        X.extend(x)
        Y.extend(y)
    X = np.dstack(X)
    X = np.rollaxis(X, -1)
    X = X.reshape(X.shape + (1,))
    random.shuffle(X)
    random.shuffle(Y)
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

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
        img = io.imread(i)
        img = cv2.resize(
            img, dsize=(192, 192), interpolation=cv2.INTER_LANCZOS4)
        images.append(img)
    y = np.concatenate((labels,))
    x = np.dstack(images)
    x = np.rollaxis(x, -1)
    x = x.reshape(x.shape + (1,))
    x = x/255.
    return x, y


def create_tensors(_final_file_names, directories):
    '''
    uses the read_images function to create the tensors and labels for train,
    validation and test dataset.
    '''
    train_paths = []
    val_paths = []
    test_paths = []
    _paths = [train_paths, val_paths, test_paths]

    for path, name, directory in zip(_paths, _final_file_names, directories):
        for f in name:
            path.append(directory+f)

    x_train, y_train = read_images(train_paths)
    x_val, y_val = read_images(val_paths)
    x_test, y_test = read_images(test_paths)

    print('Train dataset:', x_train.shape)
    print('Validation dataset:', x_val.shape)
    print('Test dataset:', x_test.shape)

    return x_train, y_train, x_val, y_val, x_test, y_test


def save_images_to_numpy(x_train, y_train, x_val, y_val, x_test, y_test):
    numpy = os.path.join(processed_data, 'numpy')
    os.makedirs(numpy, exist_ok=True)
    np.save(os.path.join(numpy, 'x_train.npy'), x_train)
    np.save(os.path.join(numpy, 'x_val.npy'), x_val)
    np.save(os.path.join(numpy, 'x_test.npy'), x_test)
    np.save(os.path.join(numpy, 'y_train.npy'), y_train)
    np.save(os.path.join(numpy, 'y_val.npy'), y_val)
    np.save(os.path.join(numpy, 'y_test.npy'), y_test)
    print('Images saved.')


def load_dataset(x, y):
    x = np.load(os.path.join(cfg.numpy, x))
    y = np.load(os.path.join(cfg.numpy, y))
    return (x, y)


if __name__ == "__main__":
    directories, file_list = get_filenames_list(cfg.processed_data)
    x_train, y_train, x_val, y_val, x_test, y_test = create_tensors(
        file_list, directories)
    save_images_to_numpy(x_train, y_train, x_val, y_val, x_test, y_test)

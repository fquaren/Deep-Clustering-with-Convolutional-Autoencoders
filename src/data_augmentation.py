from tqdm import tqdm
import os
import random
from os import listdir
from os.path import isfile, join
import imageio
import skimage
import skimage.io
import skimage.transform
from scipy import ndarray
import numpy as np


def random_rotation(image_array: ndarray):
    '''
    Pick a random degree of rotation between 25% on the left and 25% on the
    right.
    '''
    random_degree = random.uniform(-15, 15)
    return skimage.transform.rotate(image_array, random_degree)


def add_noise(image_array: ndarray):
    # sigma 0.01 default
    return skimage.util.random_noise(image_array)


def horizontal_flip(image_array: ndarray):
    '''
    Horizontal flipping of the image array.
    '''
    return image_array[:, ::-1]


def vertical_flip(image_array: ndarray):
    '''
    Vertical flipping of the image array.
    '''
    return image_array[::-1, :]


def data_aumgentation(j, processed_data, processed_dirs, train_file_names):
    '''
    To data aumgentation on train_file_names (list of image names) saving them
    in processed_data+processed_dirs[0] ('data/processed/train').
    '''
    # our folder path containing some images
    folder_path = os.path.join(processed_data, processed_dirs[0])
    # the number of file to generate
    num_files_desired = (len(train_file_names))*j
    # loop on all files of the folder and build a list of files paths
    images = [os.path.join(folder_path, f) for f in os.listdir(
        folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    available_transformations = {
        'rotate': random_rotation,
        # 'horizontal_flip': horizontal_flip,
        # 'vertical_flip': vertical_flip
    }
    for i in tqdm(range(num_files_desired)):
        # random image from the folder
        image_path = random.choice(images)
        # read image as an two dimensional array of pixels
        image_to_transform = skimage.io.imread(image_path)

        # choose a random transformation to apply for a single image
        key = random.choice(list(available_transformations))
        transformed_image = available_transformations[key](image_to_transform)

        # define a name for our new file
        image_path = image_path.split('/')[-1]
        new_file_path = '{}_augm_{}.png'.format(
            image_path.split('.')[0], i)
        transformed_image = (transformed_image*255).astype(np.uint8)
        #import pdb; pdb.set_trace()
        imageio.imwrite(
            os.path.join(folder_path, new_file_path), transformed_image)
        # write image to the disk
        # sk.io.imsave(new_file_path, transformed_image)


def check_data(
        processed_data, processed_dirs, val_file_names, test_file_names):
    '''
    Print percentage images in train, val and test.
    '''
    train_file_names = [
        f for f in listdir(os.path.join(processed_data, processed_dirs[0])) 
        if isfile(join(os.path.join(processed_data, processed_dirs[0]), f))
    ]
    # print dimensions datasets
    lenTot = len(train_file_names)+len(val_file_names)+len(test_file_names)
    percTrain = (len(train_file_names)/lenTot)*100
    percVal = (len(val_file_names)/lenTot)*100
    percTest = (len(test_file_names)/lenTot)*100
    print('Number of train images:', len(
        train_file_names), '= %.0f' % percTrain, '%')
    print('Number of validation images:', len(
        val_file_names), '= %.0f' % percVal, '%')
    print('Number of test images:', len(test_file_names), '= %.0f' % percTest, '%')

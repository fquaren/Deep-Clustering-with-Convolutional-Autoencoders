from tqdm import tqdm
import os
import random
from os import listdir
from os.path import isfile, join
import config as cfg
from glob import glob
import cv2
import numpy as np


def vertical_flip(image):
    '''
    Vertical flipping of the image array.
    '''
    image = cv2.flip(image, 0)
    return image


def horizontal_flip(image):
    '''
    Horizontal flipping of the image array.
    '''
    image = cv2.flip(image, 1)
    return image


def gaussian_noise(image):
    row, col, ch = image.shape
    mean = 0
    var = 0.5
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return noisy


def salt_and_pepper(image, prob=0.01):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def augmentation_on_train():
    '''
    To do data aumgentation on list of train images.
    '''

    folder_path = cfg.train_directory
    images = glob(os.path.join(folder_path, '*.png'))
    num_files_desired = int((len(images))/2)

    available_transformations = {
        'vertical_flip': vertical_flip,
        'horizontal_flip': horizontal_flip,
        # 'gaussian_noise': gaussian_noise,
        # 'salt_and_pepper': salt_and_pepper
    }
    for i in tqdm(range(num_files_desired)):
        image_path = random.choice(images)
        image_to_transform = cv2.imread(image_path)

        key = random.choice(list(available_transformations))
        transformed_image = available_transformations[key](image_to_transform)

        # define a name for our new file
        dir_name = os.path.dirname(image_path)
        image_name = os.path.basename(image_path).split('.')[0]
        new_file_path = os.path.join(dir_name, image_name+'_aug_'+str(i)+'.png')
        # transformed_image = (transformed_image*255).astype(np.uint8)
        cv2.imwrite(new_file_path, transformed_image)
    print('data augmentation completed. added {} images'.format(num_files_desired))


def check_data(processed_data, processed_dirs, val_file_names, test_file_names):
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


if __name__ == "__main__":
    augmentation_on_train()
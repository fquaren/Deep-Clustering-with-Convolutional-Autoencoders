import os, shutil
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split

# list of dir where the original images are stored
paths = ['../data/CT/', '../data/MRI', '../data/PET/']

# path to train, validation and test images list
file_names_path = 'data/images/'

# split parameters
train_size=0.6
val_size=0.4

# read the files from data dir
images = {
        'CT':[],
        'MRI':[],
        'PET':[]
        }

for path in paths:
    images[path] = [f for f in listdir(path) if isfile(join(path, f))]

# split images in train, validation and test lists
def split_images (train_size, val_size):
    # split train test
    CT_train, CT_test = train_test_split(images[paths[0]], train_size=train_size)
    MRI_train, MRI_test = train_test_split(images[paths[1]], train_size=train_size)
    PET_train, PET_test = train_test_split(images[paths[2]], train_size=train_size)
    # split train validation
    CT_val, CT_test = train_test_split(CT_test, train_size=val_size)
    MRI_val, MRI_test = train_test_split(MRI_test, train_size=val_size)
    PET_val, PET_test = train_test_split(PET_test, train_size=val_size)
    # create list of file names
    train_file_names = CT_train + MRI_train + PET_train
    val_file_names = CT_val + MRI_val + PET_val
    test_file_names = CT_test + MRI_test + PET_test

    #print dimensions datasets
    lenTot = len(train_file_names)+len(val_file_names)+len(test_file_names)

    return train_file_names, val_file_names, test_file_names

# save images lists
def save_list_images(images, file_names_path, file_name):
    if not os.path.exists(file_names_path):
        os.makedirs(file_names_path)
        print('Created directory for images')
    with open(file_names_path+file_name+".txt", "w") as output:
        output.write(str(images))
    print('Writing', file_name, '...')

train_images, val_images, test_images = split_images(train_size, val_size)

save_list_images(train_images, file_names_path, 'train_images')
save_list_images(val_images, file_names_path, 'val_images')
save_list_images(test_images, file_names_path, 'test_images')
print('Done, files created.')

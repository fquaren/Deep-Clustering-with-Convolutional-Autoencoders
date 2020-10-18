import os
import shutil
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from data_augmentation import data_aumgentation, check_data


def copyImages(sources, _file_names, directory):
    for source in sources:
        for f in _file_names:
            try:
                shutil.copy(os.path.join(source, f), directory)
            except OSError:
                pass
    print('Copied.')


# SETTINGS

# raw data paths
CT = '../../data/raw/CT/'
MRI = '../../data/raw/MRI/'
PET = '../../data/raw/PET/'
train_size = 0.5
val_size = 0.7

# 0.7, 0.2 senza augmentation
# 0.4, 0.4 con augmentation (j=2)
# 0.3, 0.4 con j=3
j = 2  # scale of augmentation

# ------------------------------------------------------------------------------

# IMPORT IMAGES

myPaths = [CT, MRI, PET]
myDict = {CT: [], MRI: [], PET: []}
for path in myPaths:
    myDict[path] = [f for f in listdir(path) if isfile(join(path, f))]
# Split data in train, validation and test datasets
# split train test
CT_train, CT_test = train_test_split(
    myDict[myPaths[0]], train_size=train_size)
MRI_train, MRI_test = train_test_split(
    myDict[myPaths[1]], train_size=train_size)
PET_train, PET_test = train_test_split(
    myDict[myPaths[2]], train_size=train_size)
# split train validation
CT_val, CT_test = train_test_split(CT_test, train_size=val_size)
MRI_val, MRI_test = train_test_split(MRI_test, train_size=val_size)
PET_val, PET_test = train_test_split(PET_test, train_size=val_size)
# create list of file names
train_file_names = CT_train + MRI_train + PET_train
val_file_names = CT_val + MRI_val + PET_val
test_file_names = CT_test + MRI_test + PET_test
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


# Make processed data folder and copy images in train, val and test
# delete existing processed data
processed_data = '../../data/processed'
if os.path.exists(processed_data):
    try:
        shutil.rmtree(processed_data)
    except:
        raise
# make preprocessed/
processed_dirs = [
    'train',
    'val',
    'test'
]
for d in processed_dirs:
    os.makedirs(os.path.join(processed_data, d))

# Copy images in relative dir
_file_names = [train_file_names, val_file_names, test_file_names]
for name, directory in zip(_file_names, processed_dirs):
    copyImages(myPaths, name, os.path.join(processed_data, directory))
# check
struct = ['training dataset', 'validation images', 'test images']
for d, name, s in zip(processed_dirs, _file_names, struct):
    a = len([name for name in os.listdir(os.path.join(processed_data, d))
            if os.path.isfile(os.path.join(os.path.join(processed_data, d), name))])
    if a != len(name):
        print('ERROR, i numberi non combaciano in ', s)
    else:
        print(s, 'OK.')

# ------------------------------------------------------------------------------

# DATA AUGMENTATION

data_aumgentation(j, processed_data, processed_dirs, train_file_names)
check_data(processed_data, processed_dirs, val_file_names, test_file_names)

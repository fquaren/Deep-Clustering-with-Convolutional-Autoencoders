import os
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

scans = ['CT', 'MRI', 'PET']
n_clusters = 3  # len(scans)

# Paths
processed_data = '/home/fquaren/unimib/tesi/data/processed/'
numpy = '/home/fquaren/unimib/tesi/data/processed/numpy'

CT = '/home/fquaren/unimib/tesi/data/raw/CT'
MRI = '/home/fquaren/unimib/tesi/data/raw/MRI'
PET = '/home/fquaren/unimib/tesi/data/raw/PET'
# train:    ct-2, ct-3,    mri-1, mri-3,    pet-2, pet-3
# val:      ct-4, ct-5,    mri-5, mri-6,    pet-4
# test:     ct-1,          mri-2,           pet-1
train_directory = '/home/fquaren/unimib/tesi/data/raw/train'
val_directory = '/home/fquaren/unimib/tesi/data/raw/val'
test_directory = '/home/fquaren/unimib/tesi/data/raw/test'
models = '/home/fquaren/unimib/tesi/models'
tables = '/home/fquaren/unimib/tesi/data/tables'
figures = '/home/fquaren/unimib/tesi/reports/figures'
experiments = '/home/fquaren/unimib/tesi/experiments'

exp = 'aspc_final'

ae_models = os.path.join(models, exp, 'ae')
ae_weights = os.path.join(models, exp, 'ae', 'ae_weights')
ce_weights = os.path.join(models, exp, 'ae', 'ce_weights')
final_encoder_weights = os.path.join(models, exp, 'final_encoder_weights')

# Pretrain ae settings
pretrain_epochs = 10000
ae_batch_size = 16
my_callbacks = [
    EarlyStopping(patience=25, monitor='val_loss'),
    ModelCheckpoint(
        filepath=ae_weights,
        save_best_only=True,
        save_weights_only=True,
        monitor='val_loss'
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=20,
        verbose=1,
        mode='min'
    )
]


# Pandas dataframe
d = {
    'iteration': [],
    'train_loss': [],
    'val_loss': [],
    'clustering_loss': [],
    'val_clustering_loss': [],
    'reconstruction_loss': [],
    'val_reconstruction_loss': [],
    'train_acc': [],
    'val_acc': [],
    'train_nmi': [],
    'val_nmi': [],
}

d_ae = {
    'train_loss': [],
    'val_loss': []
}

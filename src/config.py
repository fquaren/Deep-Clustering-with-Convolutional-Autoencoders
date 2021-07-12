import os
import tensorflow as tf
from keras.optimizers.schedules import PiecewiseConstantDecay, ExponentialDecay
from keras.callbacks import EarlyStopping, ModelCheckpoint
import nets
from tensorflow.keras.optimizers import Adam

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

exp = 'DCEC_12_10emb'

best_acc_ae = 0.56

cae_models = os.path.join(models, exp, 'cae')
cae_weights = os.path.join(models, exp, 'cae', 'cae_weights')
ce_weights = os.path.join(models, exp, 'cae', 'ce_weights')

# Pretrain CAE settings
cae = nets.autoencoder()

pretrain_epochs = 1000000
cae_batch_size = 12
my_callbacks = [
    EarlyStopping(patience=25, monitor='val_loss'),
    ModelCheckpoint(
        filepath=cae_weights,
        save_best_only=True,
        save_weights_only=True,
        monitor='val_loss'
    )
]
cae_optim = 'adam'

# Train DCEC settings
dcec_bs = 64
maxiter = 1000
update_interval = 10
save_interval = update_interval
tol = 0.001
gamma = 0.001
index = 0

dcec_optim = Adam(learning_rate=1e-4)

# Pandas dataframe
d = {
    'iteration': [],
    'train_loss': [],
    'val_loss': [],
    'clustering_loss': [],
    'val_clustering_loss': [],
    'reconstruction_loss': [],
    'val_reconstruction_loss': [],
    # 'train_acc': [],
    # 'val_acc': [],
    # 'train_nmi': [],
    # 'val_nmi': [],
    # 'train_ari': [],
    # 'val_ari': []
}

dec_d = {
    'iteration': [],
    # 'train_loss': [],
    # 'val_loss': [],
    'clustering_loss': [],
    'val_clustering_loss': [],
    # 'reconstruction_loss': [],
    # 'val_reconstruction_loss': [],
    # 'train_acc': [],
    # 'val_acc': [],
    # 'train_nmi': [],
    # 'val_nmi': [],
    # 'train_ari': [],
    # 'val_ari': []
}

d_cae = {
    'train_loss': [],
    'val_loss': []
}

dict_metrics = {}

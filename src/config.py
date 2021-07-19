import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
import nets

scans = ['CT', 'MRI', 'PET']
n_clusters = 3

# Paths
processed_data = '/home/fquaren/unimib/tesi/data/processed/'
numpy = '/home/fquaren/unimib/tesi/data/processed/numpy'

CT = '/home/fquaren/unimib/tesi/data/raw/CT'
MRI = '/home/fquaren/unimib/tesi/data/raw/MRI'
PET = '/home/fquaren/unimib/tesi/data/raw/PET'

train_directory = '/home/fquaren/unimib/tesi/data/raw/train'
val_directory = '/home/fquaren/unimib/tesi/data/raw/val'
test_directory = '/home/fquaren/unimib/tesi/data/raw/test'
models = '/home/fquaren/unimib/tesi/models'
tables = '/home/fquaren/unimib/tesi/data/tables'
figures = '/home/fquaren/unimib/tesi/reports/figures'
experiments = '/home/fquaren/unimib/tesi/experiments'

exp = 'DCEC_8emb'

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

# Finetuning settings
dcec_bs = 32
maxiter = 10000000
update_interval = 20
save_interval = update_interval
tol = 0.001
gamma = 0.01
index = 0
dcec_optim = 'adam'

# Pandas dataframes
d = {
    'iteration': [],
    'train_loss': [],
    'clustering_loss': [],
    'reconstruction_loss': [],
}

dec_d = {
    'iteration': [],
    'clustering_loss': [],
}

d_cae = {
    'train_loss': [],
    'val_loss': []
}

dict_metrics = {}

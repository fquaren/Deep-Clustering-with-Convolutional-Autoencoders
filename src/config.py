import os
from keras.initializers import VarianceScaling
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import nets

scans = ['CT', 'MRI', 'PET']
n_clusters = len(scans)

# Paths
processed_data = '/home/fquaren/unimib/tesi/data/processed/'
train_data = '/home/fquaren/unimib/tesi/data/processed/train'
test_data = '/home/fquaren/unimib/tesi/data/processed/test'
models = '/home/fquaren/unimib/tesi/models'
tables = '/home/fquaren/unimib/tesi/data/tables'
figures = '/home/fquaren/unimib/tesi/reports/figures'
experiments = '/home/fquaren/unimib/tesi/experiments'

exp = 'test'

cae_models = os.path.join(models, exp, 'cae')
cae_weights = os.path.join(models, exp, 'cae', 'cae_weights')
ce_weights = os.path.join(models, exp, 'cae', 'ce_weights')

# Pretrain CAE settings
cae_small = nets.CAE_Conv2DTranspose_small()
cae = nets.CAE_Conv2DTranspose_random()
init = VarianceScaling(
    scale=1./3.,
    mode='fan_in',
    distribution='uniform'
)
pretrain_epochs = 1000
cae_batch_size = 32
my_callbacks = [
    EarlyStopping(patience=10, monitor='val_loss'),
    TensorBoard(log_dir=os.path.join(experiments, exp)),
    ModelCheckpoint(
        filepath=cae_weights,
        save_best_only=True,
        save_weights_only=True,
        monitor='val_loss'
    )
]
optim = 'adam'

# Train DCEC settings
n_init_kmeans = 100
dcec_bs = 16
maxiter = 1000
update_interval = 150
save_interval = 100
tol = 0.01
gamma = 0.05
index = 0

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
    'train_ari': [],
    'val_ari': []
}

d_cae = {
    'train_loss': [],
    'val_loss': []
}

import os
from keras.optimizers.schedules import InverseTimeDecay
from keras.initializers import VarianceScaling
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import nets
from tensorflow.keras.optimizers import Adam

scans = ['CT', 'MRI', 'PET']
n_clusters = len(scans)

# Paths
processed_data = '/home/fquaren/unimib/tesi/data/processed/'
numpy = '/home/fquaren/unimib/tesi/data/processed/numpy'
train_data = '/home/fquaren/unimib/tesi/data/processed/train'
val_data = '/home/fquaren/unimib/tesi/data/processed/train'
test_data = '/home/fquaren/unimib/tesi/data/processed/test'
models = '/home/fquaren/unimib/tesi/models'
tables = '/home/fquaren/unimib/tesi/data/tables'
figures = '/home/fquaren/unimib/tesi/reports/figures'
experiments = '/home/fquaren/unimib/tesi/experiments'

exp = 'test_201120_1'

cae_models = os.path.join(models, exp, 'cae')
cae_weights = os.path.join(models, exp, 'cae', 'cae_weights')
ce_weights = os.path.join(models, exp, 'cae', 'ce_weights')

# Pretrain CAE settings
cae = nets.CAE_Conv2DTranspose()
init = VarianceScaling(
    scale=1./3.,
    mode='fan_in',
    distribution='uniform'
)
pretrain_epochs = 1000
cae_batch_size = 64
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
cae_optim = 'adam'

# Train DCEC settings
n_init_kmeans = 100
dcec_bs = 12
maxiter = 3000
update_interval = 50
save_interval = update_interval
tol = 0.1
gamma = 0.01
index = 0

initial_learning_rate = 0.001
decay_steps = 0.1
decay_rate = 0.001
learning_rate_fn = InverseTimeDecay(
    initial_learning_rate, decay_steps, decay_rate)
dcec_optim = Adam(learning_rate=learning_rate_fn)




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

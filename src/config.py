import os
from keras.initializers import VarianceScaling
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import nets

scans = ['CT', 'MRI', 'PET']
n_clusters = len(scans)

# Paths
processed_data = '/home/phil/unimib/tesi/data/processed/'
train_data = '/home/phil/unimib/tesi/data/processed/train'
test_data = '/home/phil/unimib/tesi/data/processed/test'
models = '/home/phil/unimib/tesi/models'
figures = '/home/phil/unimib/tesi/reports/figures'
experiments = '/home/phil/unimib/tesi/experiments'

exp = 'test'

cae_weights = os.path.join(models, exp,'CAE_weights')

# Pretrain settings
model = nets.CAE_Conv2DTranspose()
init = VarianceScaling(
    scale=1./3.,
    mode='fan_in',
    distribution='uniform'
)
pretrain_epochs = 1000
batch_size = 128
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
cae_loss = 'mse'

gamma = 0.1

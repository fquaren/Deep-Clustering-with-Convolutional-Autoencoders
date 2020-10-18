from keras.initializers import VarianceScaling
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import nets


# Processed data
processed_data = '/home/phil/unimib/tesi/data/processed/'

# Experiment
exp = 'test'

# Pretrain settings

net = nets.CAE_Conv2DTranspose()

init = VarianceScaling(
        scale=1./3.,
        mode='fan_in',
        distribution='uniform'
    )

pretrain_epochs = 1000
batch_size = 128

my_callbacks = [
        EarlyStopping(patience=10, monitor='val_loss'),
        TensorBoard(log_dir='./logs/'+exp),
        ModelCheckpoint(
            filepath='./models/'+exp, 
            save_best_only=True,
            save_weights_only=True,
            monitor='val_loss'
        )
    ]

optim = 'adam'
cae_loss = 'mse'

from tqdm import tqdm
import os
import numpy as np
import random
import pandas as pd
import config as cfg
from metrics import target_distribution, nmi, ari, acc
from predict import pred_dcec
from build_and_save_features import load_dataset, main
import nets
from time import time
import math
from sklearn.cluster import KMeans
from generators import generator, MyImageGenerator
from tensorflow.keras.optimizers import Adam, SGD
import visualization as viz


class ASPC(object):
    def __init__(self) -> None:
        super().__init__()
        self.autoencoder, self.encoder = nets.autoencoder()
        self.pretrained = True
        self.y_pred = []
        self.centers = []
    
    def compile(self, optimizer='sgd', loss='mse'):
        self.encoder.compile(optimizer=optimizer, loss=loss)

    def pretrain(
        self,
        autoencoder,
        encoder,
        x_train,
        train_generator,
        x_val,
        val_generator,
        batch_size,
        pretrain_epochs,
        my_callbacks,
        ae_models,
        optim
    ):
        print('pretraining...')
        encoder.summary()
        autoencoder.summary()
        autoencoder.compile(optimizer=optim, loss='mse')
        t0 = time()
        autoencoder.fit(
            train_generator,
            steps_per_epoch=math.ceil(x_train.shape[0] / batch_size),
            epochs=pretrain_epochs,
            validation_data=val_generator,
            validation_steps=math.ceil(x_val.shape[0] / batch_size),
            callbacks=my_callbacks
        ) 
        print('pretraining time: ', time() - t0)
        autoencoder.save_weights(os.path.join(ae_models, 'ae_weights'))
        encoder.save_weights(os.path.join(ae_models, 'ce_weights'))
        # save plot metrics
        cfg.d_ae['train_loss'] = autoencoder.history.history['loss']
        cfg.d_ae['val_loss'] = autoencoder.history.history['val_loss']
        print('weigths and metrics saved.')

    def init_kmeans(self, x, y):
        # Initialize model using k-means centers
        print('k-means...')
        self.encoder.load_weights(cfg.ce_weights)
        kmeans = KMeans(n_clusters=3)
        input_cluster = self.encoder.predict(x)
        y_pred = kmeans.fit_predict(input_cluster)
        centers = kmeans.cluster_centers_.astype(np.float32)
        print('metrics before training.')
        print('acc = {}; nmi = {}; ari = {}'.format(
                acc(y, y_pred),
                nmi(y, y_pred),
                ari(y, y_pred)
            )
        )
        # save metrics
        cfg.d_ae['acc'] = acc(y, y_pred)
        cfg.d_ae['nmi'] = nmi(y, y_pred)
        cfg.d_ae['ari'] = ari(y, y_pred)
        return y_pred, centers

    def update_labels(self, x, centers):
        """ Update cluster labels.
        :param x: input data, shape=(n_samples, n_features)
        :param centers: cluster centers, shape=(n_cluster, n_features)
        :return: (labels, loss): labels indicate each sample belongs to which cluster. labels[i]=j means sample i
                 belongs to cluster j; loss, the average distance between samples and their responding centers
        """
        x_norm = np.reshape(np.sum(np.square(x), 1), [-1, 1])  # column vector
        center_norm = np.reshape(np.sum(np.square(centers), 1), [1, -1])  # row vector
        dists = x_norm - 2 * np.matmul(x, np.transpose(centers)) + center_norm  # |x-y|^2 = |x|^2 -2*x*y^T + |y|^2
        labels = np.argmin(dists, 1)
        losses = np.min(dists, 1)
        return labels, losses

    def compute_sample_weight(self, losses, t, T):
        lam = np.mean(losses) + t*np.std(losses) / T
        return np.where(losses < lam, 1., 0.)

    def train(self, x_train, y_train, x_val, epochs, batch_size):
        # load weights
        self.encoder.load_weights(cfg.ce_weights)
        print('Pretrained encoder weights are loaded successfully!')

         # initialization
        t1 = time()
        self.y_pred, self.centers = self.init_kmeans(x_train, y_train)
        t2 = time()
        print('Time for initialization: %.1fs' % (t2 - t1))

        # generators
        sample_weight = np.ones(shape=x_train.shape[0])
        # define data augmentation configuration
        train_datagen = MyImageGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True, 
        )
        val_datagen = MyImageGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
        )
        # fit the data augmentation
        train_datagen.fit(x_train)
        val_datagen.fit(x_val)

        train_generator = generator(
            image_generator=train_datagen,
            x=x_train,
            y=self.centers[self.y_pred], 
            sample_weight=sample_weight,
            batch_size=batch_size
        )
        val_generator = generator(
            image_generator=val_datagen,
            x=x_val,
            batch_size=batch_size,
            shuffle=False
        )

        # training
        net_loss = 0
        clustering_loss = 0
        time_train = 0
        
        for epoch in range(epochs+1):
            t0_epoch = time()
            self.encoder.fit(
                train_generator,
                steps_per_epoch=math.ceil(x_train.shape[0] / batch_size),
                epochs=1,
                validation_data=val_generator,
                validation_steps=math.ceil(x_val.shape[0] / batch_size),
            ) 
            loss = self.encoder.history.history['loss'][0]
            val_loss = self.encoder.history.history['val_loss'][0]
            print('loss:', loss)
            print('val loss:', val_loss)

            self.y_pred, losses = self.update_labels(self.encoder.predict(x_train), self.centers)
            clustering_loss = np.mean(losses)
            print('clustering loss: ', clustering_loss)

            sample_weight = self.compute_sample_weight(losses, epoch, epochs)

            time_train = time() - t0_epoch
            print('training time:', time_train)

            if epoch % 5 == 0 or epoch == 0:
                viz.plot_cae_kmeans(
                    self.encoder, 
                    cfg.ce_weights, 
                    os.path.join(cfg.figures, cfg.exp), 
                    x_test,
                    epoch=str(epoch)
                )        
            
if __name__ == "__main__":
    # get datasets
    x_train, y_train = load_dataset('x_train.npy', 'y_train.npy')
    x_val, y_val = load_dataset('x_val.npy', 'y_val.npy')
    x_test, y_test = load_dataset('x_test.npy', 'y_test.npy')
    
    autoencoder, encoder = nets.autoencoder()
    x_train = x_train.reshape(x_train.shape[0], 128, 128, 1)
    x_val = x_val.reshape(x_val.shape[0], 128, 128, 1)
    x_test = x_test.reshape(x_test.shape[0], 128, 128, 1)

    model = ASPC()
    model.encoder.compile(optimizer=Adam(0.0001), loss='mse')
    model.encoder.summary()
    model.train(x_train=x_train, y_train=y_train, x_val=x_val, batch_size=32, epochs=10)
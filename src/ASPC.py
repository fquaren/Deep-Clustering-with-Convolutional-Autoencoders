import os
import numpy as np
import config as cfg
import metrics
from predict import init_kmeans
from build_and_save_features import load_dataset
import nets
from time import time
import math
from generators import generator, MyImageGenerator
from tensorflow.keras.optimizers import Adam
import visualization as viz


class ASPC(object):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nets.encoder()
        self.y_pred = []
        self.centers = []

    def compile(self, optimizer, loss):
        self.encoder.compile(optimizer=optimizer, loss=loss)

    def pretrain(
        self, autoencoder, encoder, x_train, train_generator, x_val,
        val_generator, batch_size, pretrain_epochs, my_callbacks,
        ae_models, optim
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

    def update_labels(self, x, centers):
        """ Update cluster labels.
        :param x: input data, shape=(n_samples, n_features)
        :param centers: cluster centers, shape=(n_cluster, n_features)
        :return: (labels, loss): labels indicate each sample belongs to which cluster. labels[i]=j means sample i
                 belongs to cluster j; loss, the average distance between samples and their responding centers
        """
        x_norm = np.reshape(np.sum(np.square(x), 1), [-1, 1])  # column vector
        center_norm = np.reshape(np.sum(np.square(centers), 1), [
                                 1, -1])  # row vector
        dists = x_norm - 2 * np.matmul(x, np.transpose(centers)) + center_norm
        # |x-y|^2 = |x|^2 -2*x*y^T + |y|^2
        labels = np.argmin(dists, 1)
        losses = np.min(dists, 1)
        return labels, losses

    def compute_sample_weight(self, losses, t, T):
        lam = np.mean(losses) + t*np.std(losses) / T
        return np.where(losses < lam, 1., 0.)

    def train(self, x_train, y_train, x_val, y_val, epochs, batch_size):
        # initialization
        t1 = time()
        self.y_pred, self.centers = init_kmeans(
            x=x_train, y=y_train, verbose=False)
        self.val_y_pred, self.val_centers = init_kmeans(
            x=x_val, y=y_val, verbose=False)
        t2 = time()
        print('Time for initialization: %.1fs' % (t2 - t1))

        # generators
        sample_weight = np.ones(shape=x_train.shape[0])
        val_sample_weight = np.ones(shape=x_val.shape[0])

        train_datagen = MyImageGenerator(
            rescale=1./225,
            featurewise_center=True,
            featurewise_std_normalization=True,
        )

        val_datagen = MyImageGenerator(
            rescale=1./225,
            featurewise_center=True,
            featurewise_std_normalization=True,
        )
        # fit the data augmentation
        train_datagen.fit(x_train)
        val_datagen.fit(x_val)

        optim = Adam(learning_rate=1e-5)
        self.encoder.compile(optimizer=optim, loss='mse')
        self.encoder.summary()

        # training
        clustering_loss = 0
        # time_train = 0
        y_pred_last = np.copy(self.y_pred)
        tol = 0.001

        # finetuning
        for epoch in range(epochs+1):
            if y_train is not None:
                acc = np.round(metrics.acc(y_train, self.y_pred), 5)
                nmi = np.round(metrics.nmi(y_train, self.y_pred), 5)
                print('ACC: {}, NMI: {}'.format(acc, nmi))

                # record the initial result
                # if epoch == 0:
                #     self.model.save_weights(save_dir + '/model_init.h5')

                # check stop criterion
                delta_y = np.sum(self.y_pred != y_pred_last).astype(
                    np.float32) / self.y_pred.shape[0]
                y_pred_last = np.copy(self.y_pred)
                if (epoch > 0 and delta_y < tol) or epoch >= epochs:
                    print('Training stopped: epoch=%d, delta_label=%.4f, tol=%.4f' % (
                        epoch, delta_y, tol))
                    # print('ASPC model saved to \'%s/model_final.h5\'' % save_dir)
                    # print('-' * 30 + ' END: time=%.1fs ' % (time()-t0) + '-' * 30)
                    self.encoder.save_weights(os.path.join(
                        cfg.ae_models, 'final_encoder_weights'))
                    # logfile.close()
                    break

            # Step 1: train the network
            self.encoder.fit(
                generator(
                    image_generator=train_datagen,
                    x=x_train,
                    y=self.centers[self.y_pred],
                    sample_weight=sample_weight,
                    batch_size=batch_size
                ),
                steps_per_epoch=math.ceil(x_train.shape[0] / batch_size),
                epochs=1,
                validation_data=generator(
                    image_generator=val_datagen,
                    x=x_val,
                    y=self.val_centers[self.val_y_pred],
                    sample_weight=val_sample_weight,
                    batch_size=batch_size,
                    shuffle=False
                ),
                validation_steps=math.ceil(x_val.shape[0] / batch_size),
                callbacks=cfg.my_callbacks
            )
            loss = self.encoder.history.history['loss'][0]
            val_loss = self.encoder.history.history['val_loss'][0]

            self.encoder.save_weights(os.path.join(
                cfg.ae_models, 'final_encoder_weights_epoch_'+str(epoch)))
            # viz.plot_ae_clusters(
            #     self.encoder,
            #     os.path.join(cfg.ae_models, 'final_encoder_weights_epoch_'+str(epoch)),
            #     os.path.join(cfg.figures, cfg.exp),
            #     x_test,
            #     epoch=str(epoch)
            # )

            # Step 2: update labels
            self.y_pred, losses = self.update_labels(
                self.encoder.predict(x_train), self.centers)
            self.val_y_pred, val_losses = self.update_labels(
                self.encoder.predict(x_val), self.val_centers)
            clustering_loss = np.mean(losses)
            val_clustering_loss = np.mean(val_losses)
            print(
                'clustering loss: ', clustering_loss,
                '\nval clustering loss: ', val_clustering_loss
            )
            
            # Step 3: Compute sample weights
            sample_weight = self.compute_sample_weight(losses, epoch, epochs)
            val_sample_weight = self.compute_sample_weight(
                val_losses, epoch, epochs)


if __name__ == "__main__":
    # get datasets
    x_train, y_train = load_dataset('x_train.npy', 'y_train.npy')
    x_val, y_val = load_dataset('x_val.npy', 'y_val.npy')
    x_test, y_test = load_dataset('x_test.npy', 'y_test.npy')

    encoder = nets.encoder()
    x_train = x_train.reshape(x_train.shape[0], 128, 128, 1)
    x_val = x_val.reshape(x_val.shape[0], 128, 128, 1)
    x_test = x_test.reshape(x_test.shape[0], 128, 128, 1)

    model = ASPC()

    model.encoder.load_weights(cfg.ce_weights)
    print('Pretrained encoder weights are loaded successfully!')

    print('initial metrics on test:')
    _, _ = init_kmeans(x=x_train, y=y_train)

    print('TRAINING')
    model.train(x_train=x_train, y_train=y_train, x_val=x_val,
                y_val=y_val, batch_size=32, epochs=1000)

    print('final metrics:')
    _, _ = init_kmeans(x=x_train, y=y_train, weights=os.path.join(
        cfg.ae_models, 'final_encoder_weights'))

    viz.plot_ae_tsne(
        encoder,
        os.path.join(cfg.ae_models, 'final_encoder_weights'),
        os.path.join(cfg.figures, cfg.exp),
        x_test
    )

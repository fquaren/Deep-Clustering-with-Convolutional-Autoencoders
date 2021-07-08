import os
import numpy as np
import config as cfg
import metrics
from predict import init_kmeans, pred_ae
from build_and_save_features import load_dataset
import nets
import math
from generators import generator, MyImageGenerator, generators
from tensorflow.keras.optimizers import Adam
import visualization as viz
import pandas as pd
from keras.callbacks import ModelCheckpoint


class ASPC(object):
    def __init__(self) -> None:
        super().__init__()
        self.autoencoder, self.encoder = nets.autoencoder()
        self.y_pred = []
        self.centers = []

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
        weights = np.where(losses < lam, 1., 0.)
        return weights

    def train(self, x_train, y_train, x_val, y_val, epochs, batch_size):
        
        # find best init random state
        for i in range(100):
            # print('final metrics:')
            _, _, _ = init_kmeans(
                x=x_train,
                x_val=x_val,
                y=y_train,
                y_val=y_val,
                random_state=i,
                weights=cfg.ae_weights,
                verbose=False
            )
            # print('RANDOM_STATE', cfg.kmeans.random_state)
            cfg.random_state_acc['acc'].append(cfg.dict_metrics['val_acc'])
            cfg.random_state_acc['nmi'].append(cfg.dict_metrics['val_nmi'])
            cfg.random_state_acc['random_state'].append(i)
        
        pass

        # initialization
        self.y_pred, self.val_y_pred, self.centers = init_kmeans(
            x=x_train,
            x_val=x_val,
            y=y_train,
            y_val=y_val,
            random_state=cfg.kmeans.random_state,
            weights=cfg.ae_weights
        )

        # weights initialization
        sample_weight = np.ones(shape=x_train.shape[0])
        sample_weight[self.y_pred == -1] = 0
        val_sample_weight = np.ones(shape=x_val.shape[0])
        val_sample_weight[self.val_y_pred == -1] = 0

        # generators
        train_datagen = MyImageGenerator(
            rescale=1./225,
            featurewise_center=True,
            featurewise_std_normalization=True,
            vertical_flip=True,
        )
        val_datagen = MyImageGenerator(
            rescale=1./225,
            featurewise_center=True,
            featurewise_std_normalization=True,
        )
        # fit the data augmentation
        train_datagen.fit(x_train)
        val_datagen.fit(x_val)

        for layer in self.encoder.layers[:-2]:
            layer.trainable = False

        optim = Adam(learning_rate=1e-5)
        self.encoder.compile(optimizer=optim, loss='mse')
        self.encoder.summary()

        # training
        clustering_loss = 0
        y_pred_last = np.copy(self.y_pred)
        tol = 0.001
        history_loss = []
        history_val_loss = []
        history_clustering_loss = []
        history_val_clustering_loss = []
        # finetuning
        for epoch in range(epochs+1):
            if y_train is not None:
                # acc = np.round(metrics.acc(y_train, self.y_pred), 5)
                # nmi = np.round(metrics.nmi(y_train, self.y_pred), 5)
                print('ACC: {}, NMI: {}'.format(
                    np.round(metrics.acc(y_train, self.y_pred), 3),
                    np.round(metrics.nmi(y_train, self.y_pred), 3)
                    )
                )

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
                    # logfile.close()
                    break

            # Step 1: train the network
            self.encoder.fit(
                generator(
                    image_generator=train_datagen,
                    x=x_train,
                    y=self.centers[self.y_pred],
                    sample_weight=sample_weight,
                    batch_size=batch_size,
                ),
                steps_per_epoch=math.ceil(x_train.shape[0] / batch_size),
                epochs=1,
                validation_data=generator(
                    image_generator=val_datagen,
                    x=x_val,
                    y=self.centers[self.val_y_pred],
                    sample_weight=val_sample_weight,
                    batch_size=batch_size,
                    shuffle=False
                ),
                validation_steps=math.ceil(x_val.shape[0] / batch_size),
                callbacks=ModelCheckpoint(
                    filepath=cfg.final_encoder_weights,
                    save_best_only=True,
                    save_weights_only=True,
                    monitor='val_loss'
                )
            )
            loss = self.encoder.history.history['loss'][0]
            val_loss = self.encoder.history.history['val_loss'][0]
            history_loss.append(loss)
            history_val_loss.append(val_loss)

            # viz.plot_ae_umap(
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
                self.encoder.predict(x_val), self.centers)
            clustering_loss = np.mean(losses)
            val_clustering_loss = np.mean(val_losses)
            print(
                'clustering loss: ', clustering_loss,
                '\nval clustering loss: ', val_clustering_loss
            )
            history_clustering_loss.append(clustering_loss)
            history_val_clustering_loss.append(val_clustering_loss)

            # # Step 3: Compute sample weights
            sample_weight = self.compute_sample_weight(losses, epoch, epochs)
            val_sample_weight = self.compute_sample_weight(val_losses, epoch, epochs)


        # Save metrics
        cfg.dict_metrics['finetuning_train_loss'] = history_loss
        cfg.dict_metrics['finetuning_val_loss'] = history_val_loss
        cfg.dict_metrics['clustering_train_loss'] = history_clustering_loss
        cfg.dict_metrics['clustering_val_loss'] = history_val_clustering_loss

        df = pd.DataFrame(data=cfg.dict_metrics)
        df.to_csv(
            os.path.join(
                cfg.tables,
                cfg.exp,
                'encoder_finetuning.csv'
            ),
            index=False
        )
        print('weigths and metrics saved.')


if __name__ == "__main__":
    # get datasets
    x_train, y_train = load_dataset('x_train.npy', 'y_train.npy')
    x_val, y_val = load_dataset('x_val.npy', 'y_val.npy')
    x_test, y_test = load_dataset('x_test.npy', 'y_test.npy')

    autoencoder, encoder = nets.autoencoder()
    x_train = x_train.reshape(x_train.shape[0], 128, 128, 1)
    x_val = x_val.reshape(x_val.shape[0], 128, 128, 1)
    x_test = x_test.reshape(x_test.shape[0], 128, 128, 1)

    method = ASPC()

    method.encoder.load_weights(cfg.ae_weights)
    print('pretrained encoder weights are loaded successfully')

    print('TRAINING')
    method.train(x_train=x_train, y_train=y_train, x_val=x_val,
                 y_val=y_val, batch_size=16, epochs=10)

    _, y_test_pred, _ = init_kmeans(
            x=x_train,
            x_val=x_test,
            y=y_train,
            y_val=y_test,
            random_state=cfg.kmeans.random_state,
            weights=cfg.final_encoder_weights,
        )
    
    # for i in range(100):
    #     # print('final metrics:')
    #     _, y_test_pred, _ = init_kmeans(
    #         x=x_train, x_val=x_test, y=y_train, y_val=y_test, random_state=i, weights=cfg.final_encoder_weights, verbose=False)
    #     # print('RANDOM_STATE', cfg.kmeans.random_state)
    #     cfg.random_state_acc['test_acc'].append(cfg.dict_metrics['val_acc'])
    #     cfg.random_state_acc['test_nmi'].append(cfg.dict_metrics['val_nmi'])
    #     cfg.random_state_acc['random_state'].append(i)

    # df = pd.DataFrame(data=cfg.random_state_acc)
    # df.to_csv(
    #     os.path.join(
    #         cfg.tables,
    #         cfg.exp,
    #         'random_state_acc.csv'
    #     ),
    #     index=False
    # )

    viz.plot_ae_tsne(
        encoder,
        cfg.final_encoder_weights,
        os.path.join(cfg.figures, cfg.exp),
        x_train,
        x_test
    )
    viz.plot_ae_umap(
        encoder,
        cfg.final_encoder_weights,
        os.path.join(cfg.figures, cfg.exp),
        x_train,
        x_test
    )

    viz.plot_confusion_matrix(y_test, y_test_pred)

    # viz.feature_map(scan=cfg.scans[0], exp=cfg.exp, layer=1, depth=32, weights=cfg.final_encoder_weights)
    # viz.feature_map(scan=cfg.scans[1], exp=cfg.exp, layer=1, depth=32, weights=cfg.final_encoder_weights)
    # viz.feature_map(scan=cfg.scans[2], exp=cfg.exp, layer=1, depth=32, weights=cfg.final_encoder_weights)
    # viz.feature_map(scan=cfg.scans[0], exp=cfg.exp, layer=2, depth=64, weights=cfg.final_encoder_weights)
    # viz.feature_map(scan=cfg.scans[1], exp=cfg.exp, layer=2, depth=64, weights=cfg.final_encoder_weights)
    # viz.feature_map(scan=cfg.scans[2], exp=cfg.exp, layer=2, depth=64, weights=cfg.final_encoder_weights)

import os
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from keras.models import Model
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment as linear_assignment
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import config as cfg
from nets import ClusteringLayer
from build_and_save_features import load_dataset
from metrics import acc, nmi, ari


def plot_cae_tnse(autoencoder, encoder, models_directory, figures, dataset):
    """
    parameters:
    - autoencoder,
    - encoder,
    - models_directory: directory containing the models,
    - figures: directory to save the plots
    - dataset: dataset on which to predict (train, val, test)

    Loads the model weigths from models directory, predicts model output,
    perfoms kmeans and tsne and plots result.
    """
    autoencoder.load_weights(os.path.join(models_directory, 'cae_weights'))
    kmeans = KMeans(n_clusters=cfg.n_clusters, n_init=50)
    features = encoder.predict(dataset)[1]
    y_pred = kmeans.fit_predict(features)
    tsne = TSNE(n_components=2, perplexity=50)
    embedding = tsne.fit_transform(features)
    plt.figure()
    plt.scatter(embedding[:, 0], embedding[:, 1], c=y_pred, s=20, cmap='brg')
    plt.savefig(os.path.join(figures, 'tsne_cae'))
    print('saved scatter plot cae')


def plot_pretrain_metrics(file, save_dir):
    '''
    This function read a csv file containing the pretraining metrics, plots
    them and saves an image in the figures folder.
    '''
    data = pd.read_csv(file)
    train_loss = data['train_loss']
    val_loss = data['val_loss']
    plt.figure()
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training loss', 'Validation loss'])
    plt.savefig(os.path.join(save_dir, 'pretrain_metrics'))


def plot_dcec_tsne(model, models_directory, figures, dataset):
    """
    parameters:
    - model: keras model,
    - models_directory: directory containing the models,
    - figures: directory to save the plots
    - dataset: dataset on which to predict (train, val, test)

    Loads the model weigths from models directory, predicts model output,
    perfoms kmeans and tsne and plots result.
    """
    for model_name in tqdm(os.listdir(models_directory)):
        ite = model_name.split('_')[2].split('.')[0]
        model.load_weights(os.path.join(models_directory, model_name))
        kmeans = KMeans(n_clusters=cfg.n_clusters, n_init=50)
        features = model.predict(dataset)[0]
        y_pred = kmeans.fit_predict(features)
        tsne = TSNE(n_components=2, perplexity=50)
        embedding = tsne.fit_transform(features)
        plt.figure()
        plt.scatter(
            embedding[:, 0], embedding[:, 1], c=y_pred, s=20, cmap='brg')
        plt.savefig(os.path.join(figures, 'tsne_{}'.format(ite)))
        print('saved scatter plot ite_{}'.format(ite))


def plot_train_metrics(file, save_dir):
    '''
    This function read a csv file containing the training metrics, plots them
    and saves an image in the figures folder.
    '''
    data = pd.read_csv(file)

    ite = data['iteration']
    train_loss = data['train_loss']
    val_loss = data['val_loss']
    clust_loss = data['clustering_loss']
    val_clust_loss = data['val_clustering_loss']
    rec_loss = data['reconstruction_loss']
    val_rec_loss = data['val_reconstruction_loss']
    train_acc = data['train_acc']
    val_acc = data['val_acc']
    train_nmi = data['train_nmi']
    val_nmi = data['val_nmi']
    train_ari = data['train_ari']
    val_ari = data['val_ari']

    # losses
    plt.figure(figsize=(30, 10))
    plt.subplot(1, 3, 1)
    x1 = ite
    y1 = train_loss
    y2 = val_loss
    plt.plot(x1, y1)
    plt.plot(x1, y2)
    plt.title('L')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])

    plt.subplot(1, 3, 2)
    x1 = ite
    y1 = clust_loss
    y2 = val_clust_loss
    plt.plot(x1, y1)
    plt.plot(x1, y2)
    plt.title('Lc: clustering loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])

    plt.subplot(1, 3, 3)
    x1 = ite
    y1 = rec_loss
    y2 = val_rec_loss
    plt.plot(x1, y1)
    plt.plot(x1, y2)
    plt.title('Lr: reconstruction loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.savefig(os.path.join(save_dir, 'train_val_loss'))

    # other metrics
    plt.figure(figsize=(30, 10))
    plt.subplot(1, 3, 1)
    x1 = ite
    y1 = train_acc
    y2 = val_acc
    plt.plot(x1, y1)
    plt.plot(x1, y2)
    plt.title('Accuracy')
    plt.ylabel('Acc')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])

    plt.subplot(1, 3, 2)
    x1 = ite
    y1 = train_nmi
    y2 = val_nmi
    plt.plot(x1, y1)
    plt.plot(x1, y2)
    plt.title('NMI')
    plt.ylabel('NMI')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])

    plt.subplot(1, 3, 3)
    x1 = ite
    y1 = train_ari
    y2 = val_ari
    plt.plot(x1, y1)
    plt.plot(x1, y2)
    plt.title('ARI')
    plt.ylabel('ARI')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.savefig(os.path.join(save_dir, 'train_val_acc_nmi_ari'))


def test_dcec(model, x, y):
    metrics = {}
    test_q, _ = model.predict(x, verbose=0)
    # test_p = target_distribution(test_q)
    # test_loss = model.fit(x=x, y=[test_p, x], verbose=0)
    y_test_pred = test_q.argmax(1)
    metrics['test_acc'] = acc(y, y_test_pred)
    metrics['test_nmi'] = nmi(y, y_test_pred)
    metrics['test_ari'] = ari(y, y_test_pred)
    return metrics, y_test_pred


def plot_confusion_matrix(y_true, y_pred, save_dir):
    matrix = confusion_matrix(
        [int(i) for i in y_true], y_pred)

    plt.figure()
    sns.heatmap(matrix, annot=True, fmt="d", annot_kws={"size": 20})
    plt.title("Confusion matrix")
    plt.ylabel('True label')
    plt.xlabel('Clustering label')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix'))

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    # Confusion matrix.
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(-w)
    a = ind[0].tolist()
    b = ind[1].tolist()
    print(np.array([a, b]))


if __name__ == "__main__":
    x_test, y_test = load_dataset('x_test.npy', 'y_test.npy')

    cae, encoder = cfg.cae
    clustering_layer = ClusteringLayer(
        cfg.n_clusters, name='clustering')(encoder.output[1])
    model = Model(
        inputs=encoder.input, outputs=[clustering_layer, cae.output])
    model.compile(
        loss=['kld', 'mse'], loss_weights=[cfg.gamma, 1], optimizer='adam')

    os.makedirs(os.path.join(cfg.figures, cfg.exp, 'cae'), exist_ok=True)
    os.makedirs(os.path.join(cfg.figures, cfg.exp, 'dcec'), exist_ok=True)
    # --- CAE ---
    # plot tsne after kmean init
    plot_cae_tnse(
        autoencoder=cae,
        encoder=encoder,
        models_directory=os.path.join(cfg.models, cfg.exp, 'cae'),
        figures=os.path.join(cfg.figures, cfg.exp, 'cae'),
        dataset=x_test
    )

    # plot pretrain metrics
    plot_pretrain_metrics(
        file=os.path.join(cfg.tables, 'cae_train_metrics.csv'),
        save_dir=os.path.join(cfg.figures, cfg.exp, 'cae'),
    )

    # --- DCEC ---
    # plot tsne dcec iterations during training
    # plot_dcec_tsne(
    #     model=model,
    #     models_directory=os.path.join(cfg.models, cfg.exp, 'dcec'),
    #     figures=os.path.join(cfg.figures, cfg.exp, 'dcec'),
    #     dataset=x_test
    # )

    # plot train metrics
    plot_train_metrics(
        file=os.path.join(cfg.tables, cfg.exp, 'dcec_train_metrics.csv'),
        save_dir=os.path.join(cfg.figures, cfg.exp, 'dcec')
    )

    metrics, y_pred = test_dcec(model, x_test, y_test)
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        save_dir=os.path.join(cfg.figures, cfg.exp, 'dcec')
    )
    print('final metrics:', metrics)

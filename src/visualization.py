import os
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from keras.models import Model
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment as linear_assignment
import seaborn as sns
import pandas as pd
import config as cfg
from build_and_save_features import load_dataset
import nets
import umap
import predict
import random
import math
from scipy.spatial import Voronoi


def plot_ae_tsne(encoder, weights, figures, dataset, test_dataset, epoch=''):
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
    encoder.load_weights(weights)
    kmeans = cfg.kmeans
    features = encoder.predict(dataset)
    _ = kmeans.fit_predict(features)
    test_features = encoder.predict(test_dataset)
    y_test_pred = kmeans.predict(test_features)
    plt.figure()
    tsne = TSNE(n_components=2, perplexity=50, n_iter=3000)
    embedding = tsne.fit_transform(test_features)
    plt.scatter(embedding[:, 0], embedding[:, 1],
                c=y_test_pred, s=20, cmap='brg')
    plt.savefig(os.path.join(figures, 'tsne_encoder_' + epoch))

    print('saved scatter plot ae')


def plot_ae_umap(encoder, weights, figures, train_dataset, dataset, epoch=''):
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
    encoder.load_weights(weights)
    kmeans = cfg.kmeans
    features = encoder.predict(train_dataset)
    _ = kmeans.fit_predict(features)
    centers = kmeans.cluster_centers_.astype(np.float32)
    test_features = encoder.predict(dataset)
    y_test_pred = kmeans.predict(test_features)
    reducer = umap.UMAP()
    reducer.fit(features)
    test_embedding = reducer.transform(test_features)
    centers2d = reducer.transform(centers)
    min_x = min(test_embedding, key=lambda x: x[0])[0]
    min_y = min(test_embedding, key=lambda x: x[1])[1]
    max_x = max(test_embedding, key=lambda x: x[0])[0]
    max_y = max(test_embedding, key=lambda x: x[1])[1]
    centers2d = np.append(
        centers2d, [[999, 999], [-999, 999], [999, -999], [-999, -999]], axis=0)

    vor = Voronoi(centers2d)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    # colorize
    for region in regions:
        polygon = vertices[region]
        plt.fill(*zip(*polygon), alpha=0.4)
    plt.scatter(test_embedding[:, 0], test_embedding[:,1], c=y_test_pred, s=20, cmap='brg')
    plt.scatter(centers2d[:, 0], centers2d[:, 1], c='black', s=100)
    plt.xlim((min_x - 1, max_x + 1))
    plt.ylim((min_y - 1, max_y + 1))
    plt.savefig(os.path.join(figures, 'umap_encoder_' + epoch))
    print('saved scatter plot ae')


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def plot_pretrain_metrics(file, save_dir):
    '''
    This function reads a csv file containing the pretraining metrics, plots
    them and saves an image in the figures folder.
    '''
    data = pd.read_csv(file)
    train_loss = data['pretrain_train_loss']
    val_loss = data['pretrain_val_loss']
    plt.figure()
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('Pretrain loss')
    plt.xlabel('Epoch')
    plt.legend(['Training loss', 'Validation loss'])
    plt.savefig(os.path.join(save_dir, 'pretrain_metrics'))


def plot_finetuning_losses(file, save_dir):
    '''
    This function read a csv file containing the training metrics, plots them
    and saves an image in the figures folder.
    '''
    data = pd.read_csv(file)

    train_loss = data['finetuning_train_loss']
    val_loss = data['finetuning_val_loss']
    clust_loss = data['clustering_train_loss']
    val_clust_loss = data['clustering_val_loss']

    ite = range(len(train_loss))

    # losses
    plt.figure()
    plt.subplot(1, 2, 1)
    x1 = ite
    y1 = train_loss
    y2 = val_loss
    plt.plot(x1, y1)
    plt.plot(x1, y2)
    plt.title('L')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])

    plt.subplot(1, 2, 2)
    x1 = ite
    y1 = clust_loss
    y2 = val_clust_loss
    plt.plot(x1, y1)
    plt.plot(x1, y2)
    plt.title('Clustering loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.savefig(os.path.join(save_dir, 'finetuning_loss.svg'))


def plot_metrics(file, save_dir):
    '''
    This function read a csv file containing the training metrics, plots them
    and saves an image in the figures folder.
    '''
    data = pd.read_csv(file)

    train_acc = data['acc']
    train_nmi = data['nmi']

    mean_train_acc = np.mean(train_acc)
    mean_train_nmi = np.mean(train_nmi)
    std_train_acc = np.std(train_acc)
    std_train_nmi = np.std(train_nmi)

    plt.subplot(1, 2, 1)
    plt.errorbar(1, mean_train_acc, yerr=std_train_acc, fmt='o')
    plt.title('Pre finetuting accuracy')
    plt.ylabel('ACC')
    plt.legend('Train')

    plt.subplot(1, 2, 2)
    plt.errorbar(1, mean_train_nmi, yerr=std_train_nmi, fmt='o')
    plt.title('Pre finetuting NMI')
    plt.ylabel('NMI')
    plt.legend('Train')
    plt.savefig(os.path.join(save_dir))


def plot_confusion_matrix(y_true, y_pred, save_dir=os.path.join(cfg.figures, cfg.exp)):
    matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)

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


def plot_dataset():
    # plots some random images from train directory
    imgs = []
    for i in range(40):
        for scan in cfg.scans:
            n = random.randint(0, 50)
            imgs.append(predict.get_image(
                predict.get_list_per_type(cfg.train_directory, scan), n))
    random.shuffle(imgs)

    fig = plt.figure(frameon=False, figsize=(100, 100))

    k = 10
    columns = k
    rows = k
    ax = []
    for i in range(1, columns*rows + 1):
        img = imgs[i]
        ax.append(fig.add_subplot(rows, columns, i))
        plt.imshow(img)
        plt.axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0, left=0, right=1, bottom=0, top=1)

    os.makedirs(os.path.join(cfg.figures, 'dataset'), exist_ok=True)
    plt.savefig(os.path.join(cfg.figures, 'dataset', 'scans.svg'))


def feature_map(scan, layer, depth, exp, weights):
    # load image
    # load network aspc_29_CAE
    encoder = nets.encoder()
    encoder.load_weights(weights)
    model = Model(inputs=encoder.inputs, outputs=encoder.layers[layer].output)
    img = predict.get_image(
        predict.get_list_per_type(cfg.train_directory, scan), 1)
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    feature_maps = model.predict(img)

    plt.figure(frameon=False, figsize=(30, 30))

    # plot all
    square = math.sqrt(depth)
    if isinstance(square, float):
        square = int(square + 1)
    ix = 1
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            plt.subplot(square, square, ix)
            # plot filter channel in grayscale
            try:
                plt.imshow(feature_maps[0, :, :, ix-1])
            except:
                plt.imshow(np.zeros((feature_maps.shape[1], feature_maps.shape[1]), dtype=np.uint8))
            plt.axis('off')
            ix += 1
        plt.subplots_adjust(wspace=0.1, hspace=0, left=0,
                            right=1, bottom=0, top=1)
    # show the figure

    os.makedirs(os.path.join(cfg.figures, cfg.exp,
                             'feature_maps'), exist_ok=True)
    plt.savefig(os.path.join(cfg.figures, cfg.exp, 'feature_maps',
                             'conv_layer_' + scan + '_' + str(layer) + '.svg'))


if __name__ == "__main__":
    x_train, y_train = load_dataset('x_train.npy', 'y_train.npy')
    x_test, y_test = load_dataset('x_test.npy', 'y_test.npy')

    feature_map(scan=cfg.scans[0], exp='aspc_29_CAE', layer=1, depth=32)
    feature_map(scan=cfg.scans[1], exp='aspc_29_CAE', layer=1, depth=32)
    feature_map(scan=cfg.scans[2], exp='aspc_29_CAE', layer=1, depth=32)
    feature_map(scan=cfg.scans[0], exp='aspc_29_CAE', layer=2, depth=64)
    feature_map(scan=cfg.scans[1], exp='aspc_29_CAE', layer=2, depth=64)
    feature_map(scan=cfg.scans[2], exp='aspc_29_CAE', layer=2, depth=64)

    # autoencoder, encoder = nets.autoencoder(x_test)

    # plot_cae_kmeans(
    #     encoder,
    #     cfg.ce_weights,
    #     os.path.join(cfg.figures, cfg.exp, 'cae'),
    #     x_test
    # )

    # clustering_layer = ClusteringLayer(
    #     cfg.n_clusters, name='clustering')(encoder.output)
    # model = Model(
    #     inputs=encoder.input, outputs=[clustering_layer, cae.output])
    # model.compile(
    #     loss=['kld', 'mse'], loss_weights=[cfg.gamma, 1], optimizer='adam')

    # os.makedirs(os.path.join(cfg.figures, cfg.exp, 'cae'), exist_ok=True)
    # os.makedirs(os.path.join(cfg.figures, cfg.exp, 'dcec'), exist_ok=True)
    # # --- CAE ---
    # # plot tsne after kmean init
    # plot_cae_tnse(
    #     autoencoder=cae,
    #     encoder=encoder,
    #     models_directory=os.path.join(cfg.models, cfg.exp, 'cae'),
    #     figures=os.path.join(cfg.figures, cfg.exp, 'cae'),
    #     dataset=x_test
    # )

    # # plot pretrain metrics
    # plot_pretrain_metrics(
    #     file=os.path.join(cfg.tables, 'cae_train_metrics.csv'),
    #     save_dir=os.path.join(cfg.figures, cfg.exp, 'cae'),
    # )

    # # --- DCEC ---
    # # plot tsne dcec iterations during training
    # plot_dcec_tsne(
    #     model=model,
    #     models_directory=os.path.join(cfg.models, cfg.exp, 'dcec'),
    #     figures=os.path.join(cfg.figures, cfg.exp, 'dcec'),
    #     dataset=x_test
    # )

    # # plot train metrics
    # plot_train_metrics(
    #     file=os.path.join(cfg.tables, cfg.exp, 'dcec_train_metrics.csv'),
    #     save_dir=os.path.join(cfg.figures, cfg.exp, 'dcec')
    # )

    # metrics, y_pred = test_dcec(model, x_test, y_test)
    # plot_confusion_matrix(
    #     y_true=y_test,
    #     y_pred=y_pred,
    #     save_dir=os.path.join(cfg.figures, cfg.exp, 'dcec')
    # )
    # print('final metrics:', metrics)

# TODO https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/

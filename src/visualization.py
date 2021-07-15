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
import umap
from scipy.spatial import Voronoi
import nets
import predict
import math


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
    kmeans = KMeans(n_clusters=3, n_init=100)
    features = encoder.predict(dataset)
    _ = kmeans.fit_predict(features)
    test_features = encoder.predict(test_dataset)
    y_test_pred = kmeans.predict(test_features)
    plt.figure()
    tsne = TSNE(n_components=2)
    embedding = tsne.fit_transform(test_features)
    plt.scatter(embedding[:, 0], embedding[:, 1],
                c=y_test_pred, s=20, cmap='brg')
    plt.savefig(os.path.join(figures, 'tsne_encoder_' + epoch + '.svg'))

    plt.close()

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
    kmeans = kmeans = KMeans(n_clusters=3, n_init=100)
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
        centers2d,
        [[999, 999], [-999, 999], [999, -999], [-999, -999]],
        axis=0
    )

    vor = Voronoi(centers2d)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    # colorize
    plt.figure()
    for region in regions:
        polygon = vertices[region]
        plt.fill(*zip(*polygon), alpha=0.4)
    plt.scatter(
        test_embedding[:, 0],
        test_embedding[:, 1],
        c=y_test_pred,
        s=20,
        cmap='brg'
    )
    plt.scatter(centers2d[:, 0], centers2d[:, 1], c='black', s=100)
    plt.xlim((min_x - 1, max_x + 1))
    plt.ylim((min_y - 1, max_y + 1))
    # os.makedirs(os.path.join(cfg.figures, cfg.exp, 'umap'), exist_ok=True)
    plt.savefig(os.path.join(figures, 'umap_encoder_' + epoch + '.svg'))
    plt.close()
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
    train_loss = data['train_loss']
    val_loss = data['val_loss']
    plt.figure()
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('Pretrain loss')
    plt.xlabel('Epoch')
    plt.legend(['Training loss', 'Validation loss'])
    plt.savefig(os.path.join(save_dir, 'pretrain_metrics.svg'))


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
        tsne = TSNE(n_components=2)
        embedding = tsne.fit_transform(features)
        plt.figure()
        plt.scatter(
            embedding[:, 0], embedding[:, 1], c=y_pred, s=20, cmap='brg')
        plt.savefig(os.path.join(figures, 'tsne_{}'.format(ite)))
        print('saved scatter plot ite_{}'.format(ite))


def plot_confusion_matrix(
    y_true,
    y_pred,
    save_dir=os.path.join(cfg.figures, cfg.exp)
):
    matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)

    plt.figure()
    sns.heatmap(matrix, annot=True, fmt="d", annot_kws={"size": 20})
    plt.title("Confusion matrix")
    plt.ylabel('True label')
    plt.xlabel('Clustering label')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.svg'))

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    # Confusion matrix.
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(-w)
    a = ind[0].tolist()
    b = ind[1].tolist()
    print(np.array([a, b]))
    plt.close()


def feature_map(scan, layer, depth, exp, weights):
    encoder = nets.encoder()
    encoder.load_weights(weights)
    model = Model(inputs=encoder.inputs, outputs=encoder.layers[layer].output)

    # load image
    img = predict.get_image(
        predict.get_list_per_type(cfg.train_directory, scan), 1)
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    # get prediction
    feature_maps = model.predict(img)

    # immagine da plottare
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
                plt.imshow(
                    np.zeros(
                        (feature_maps.shape[1], feature_maps.shape[1]),
                        dtype=np.uint8
                        )
                    )
            plt.axis('off')
            ix += 1
        plt.subplots_adjust(wspace=0.1, hspace=0, left=0,
                            right=1, bottom=0, top=1)

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
                plt.imshow(
                    np.zeros(
                        (feature_maps.shape[1], feature_maps.shape[1]),
                        dtype=np.uint8
                    )
                )
            plt.axis('off')
            ix += 1
        plt.subplots_adjust(wspace=0.1, hspace=0, left=0,
                            right=1, bottom=0, top=1)
    # show the figure

    os.makedirs(os.path.join(cfg.figures, cfg.exp,
                             'feature_maps'), exist_ok=True)
    plt.savefig(os.path.join(cfg.figures, cfg.exp, 'feature_maps',
                             'conv_layer_' + scan + '_' + str(layer) + '.svg'))
    plt.close()


if __name__ == "__main__":
    pass

import os
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from keras.models import Model
from matplotlib import pyplot as plt
from tqdm import tqdm
import config as cfg
from nets import CAE_Conv2DTranspose, ClusteringLayer
from build_features import get_filenames_list, create_tensors


def plot_tsne_ite(model, models_directory, figures, dataset):
    """
    parameters:
    - model: keras model,
    - models_directory: directory containing the models,
    - figures: directory to save the plots
    - dataset: dataset on which to predict (train, val, test)

    Loads the model weigths from models directory, predicts model output,
    perfoms kmeans and tsne and plots result.
    """
    for model_name in tqdm(models_directory):
        if 'dcec' in model_name:
            ite = model_name.split('_')[2].split('.')[0]
        else:
            ite = 'cae'
        model.load_weights(os.path.join(models_directory, model_name))
        kmeans = KMeans(n_clusters=cfg.n_clusters, n_init=50)
        features = model.predict(dataset)[0]
        y_pred = kmeans.fit_predict(features)
        tsne = TSNE(n_components=2, perplexity=50)
        embedding = tsne.fit_transform(features)
        plt.figure()
        plt.scatter(embedding[:, 0], embedding[:, 1], c=y_pred, s=20, cmap='brg')
        plt.savefig(os.path.join(figures, 'tsne_{}'.format(ite)))
        print('saved scatter plot ite_{}'.format(ite))


if __name__ == "__main__":
    directories, file_list = get_filenames_list(cfg.processed_data)
    x_train, y_train, x_val, y_val, x_test, y_test = create_tensors(
        file_list, directories)

    cae, encoder = CAE_Conv2DTranspose()
    clustering_layer = ClusteringLayer(
        cfg.n_clusters, name='clustering')(encoder.output[1])
    model = Model(
        inputs=encoder.input, outputs=[clustering_layer, cae.output])
    model.compile(loss=['kld', 'mse'], loss_weights=[0.1, 1], optimizer='adam')

    # plot tsne after kmean init
    plot_tsne_ite(
        encoder,
        os.listdir(os.path.join(cfg.models, cfg.exp, 'cae')),
        os.path.join(cfg.figures, cfg.exp),
        x_train
    )

    # plot tsne dcec iterations during training
    plot_tsne_ite(
        model,
        os.listdir(os.path.join(cfg.models, cfg.exp, 'dcec')),
        os.path.join(cfg.figures, cfg.exp),
        x_train
        )

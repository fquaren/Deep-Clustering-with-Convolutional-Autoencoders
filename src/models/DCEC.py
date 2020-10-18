
from keras.models import Model
from nets import CAE_Conv2DTranspose
from clustering_layer import ClusteringLayer
import config as cfg  # TODO fix


n_clusters = cfg.n_clusters

autoencoder, encoder = CAE_Conv2DTranspose()

clustering_layer = ClusteringLayer(
    n_clusters, name='clustering')(encoder.output[1])
model = Model(
    inputs=encoder.input, outputs=[clustering_layer, autoencoder.output])

# import pdb; pdb.set_trace()

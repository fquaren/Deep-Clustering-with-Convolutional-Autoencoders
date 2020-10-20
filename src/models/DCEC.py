
from keras.models import Model
from clustering_layer import ClusteringLayer
import sys
sys.path.append('/home/phil/unimib/tesi/src')
from nets import CAE_Conv2DTranspose
import config as cfg

n_clusters = cfg.n_clusters

autoencoder, encoder = CAE_Conv2DTranspose()

clustering_layer = ClusteringLayer(
    n_clusters, name='clustering')(encoder.output[1])
model = Model(
    inputs=encoder.input, outputs=[clustering_layer, autoencoder.output])

# import pdb; pdb.set_trace()

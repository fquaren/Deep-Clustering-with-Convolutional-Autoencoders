from keras.layers import Input, Dense, Conv2D, Flatten, Reshape, Conv2DTranspose, BatchNormalization, Concatenate
from keras.models import Model
from keras import backend as K
from keras.engine.topology import InputSpec, Layer
from tensorflow.keras import initializers


def CAE_Conv2DTranspose(input_shape=(192, 192, 1), filters=[16, 32, 64, 256, 3]):

    input_img = Input(shape=input_shape)

    # Encoder
    x = Conv2D(filters[0], 5, strides=3, padding='same', activation='relu', name='conv1', input_shape=input_shape)(input_img)
    x = Conv2D(filters[1], 3, strides=2, padding='same', activation='relu', name='conv2')(x)
    x = Conv2D(filters[2], 3, strides=2, padding='same', activation='relu', name='conv3')(x)


    x = Flatten(name='flatten_1')(x)

    encoded = Dense(units=filters[-2], activation='relu', name='encoded')(x)
    
    y = Dense(
        units=filters[-1], 
        # kernel_initializer=initializers.VarianceScaling(scale=0.1, mode="fan_out", distribution="uniform"),
        # bias_initializer=initializers.Zeros(),
        activation='relu', 
        name='input_clustering'
    )(encoded)

    # Decoder
    x = Dense(units=16*16*filters[2], activation='relu')(encoded)
    x = Reshape((16, 16, filters[2]))(x)
    x = Conv2DTranspose(filters[1], 3, strides=2, padding='same', activation='relu', name='deconv3')(x)
    x = Conv2DTranspose(filters[0], 3, strides=2, padding='same', activation='relu', name='deconv2')(x)
    decoded = Conv2DTranspose(1, 5, strides=3, padding='same', name='deconv1')(x)

    return Model(inputs=input_img, outputs=decoded, name='CAE'), Model(inputs=input_img, outputs=[encoded, y], name='CE')


def CAE_Conv2DTranspose_old(input_shape=(192, 192, 1), filters=[16, 32, 64, 3]):

    input_img = Input(shape=input_shape)

    # Encoder
    x = Conv2D(filters[0], 3, strides=1, padding='same', activation='relu', name='conv1', input_shape=input_shape)(input_img)
    x = Conv2D(filters[1], 7, strides=4, padding='same', activation='relu', name='conv2')(x)
    x = Conv2D(filters[2], 7, strides=4, padding='same', activation='relu', name='conv3')(x)
    x = Flatten(name='flatten_1')(x)

    encoded = Dense(units=filters[-1], name='embedding')(x)

    # Decoder
    x = Dense(units=4*4*filters[2], activation='relu')(encoded)
    x = Reshape((4, 4, filters[2]))(x)
    x = Conv2DTranspose(filters[1], 7, strides=4, padding='same', activation='relu', name='deconv3')(x)
    x = Conv2DTranspose(filters[0], 7, strides=4, padding='same', activation='relu', name='deconv2')(x)
    decoded = Conv2DTranspose(1, 3, strides=1, padding='same', name='deconv1')(x)

    return Model(inputs=input_img, outputs=decoded, name='CAE'), Model(inputs=input_img, outputs=encoded, name='CE')


def CAE_Conv2DTranspose_small(input_shape=(192, 192, 1), filters=[16, 32, 3]):

    input_img = Input(shape=input_shape)

    # Encoder
    x = Conv2D(filters[0], 5, strides=3, padding='same', activation='relu', name='conv1', input_shape=input_shape)(input_img)
    x = Conv2D(filters[1], 7, strides=4, padding='same', activation='relu', name='conv2')(x)
    x = Flatten(name='flatten_1')(x)
    
    encoded = Dense(units=filters[-1], name='embedding')(x)

    # Decoder
    x = Dense(units=16*16*filters[1], activation='relu')(encoded)
    x = Reshape((16, 16, filters[1]))(x)
    x = Conv2DTranspose(filters[0], 7, strides=4, padding='same', activation='relu', name='deconv2')(x)
    decoded = Conv2DTranspose(1, 5, strides=3, padding='same', name='deconv1')(x)

    return Model(inputs=input_img, outputs=decoded, name='CAE'), Model(inputs=input_img, outputs=encoded, name='CE') 


def CAE_Conv2DTranspose_big(input_shape=(192, 192, 1), filters=[16, 32, 3]):

    input_img = Input(shape=input_shape)

    # Encoder
    x = Conv2D(filters[0], 3, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape)(input_img)
    x = Conv2D(filters[1], 3, strides=2, padding='same', activation='relu', name='conv2')(x)
    x = Flatten(name='flatten_1')(x)

    encoded = Dense(units=filters[-1], name='embedding')(x)

    # Decoder
    x = Dense(units=48*48*filters[1], activation='relu')(x)
    x = Reshape((48, 48, filters[1]))(x)
    x = Conv2DTranspose(filters[0], 3, strides=2, padding='same', activation='relu', name='deconv2')(x)
    decoded = Conv2DTranspose(1, 3, strides=2, padding='same', name='deconv1')(x)

    return Model(inputs=input_img, outputs=decoded, name='CAE'), Model(inputs=input_img, outputs=[encoded], name='CE')


def CAE_Conv2DTranspose_big_dense(input_shape=(192, 192, 1), filters=[16, 32, 300, 3]):

    input_img = Input(shape=input_shape)

    # Encoder
    x = Conv2D(filters[0], 3, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape)(input_img)
    x = Conv2D(filters[1], 3, strides=2, padding='same', activation='relu', name='conv2')(x)
    x = Flatten(name='flatten_1')(x)
    x = Dense(units=filters[-2], name='dense1')(x)

    encoded = Dense(units=filters[-1], name='embedding')(x)

    # Decoder
    x = Dense(units=filters[-2], name='dense2', activation='relu')(encoded)
    x = Dense(units=48*48*filters[1], activation='relu')(x)
    x = Reshape((48, 48, filters[1]))(x)
    x = Conv2DTranspose(filters[0], 3, strides=2, padding='same', activation='relu', name='deconv2')(x)
    decoded = Conv2DTranspose(1, 3, strides=2, padding='same', name='deconv1')(x)

    return Model(inputs=input_img, outputs=decoded, name='CAE'), Model(inputs=input_img, outputs=[encoded], name='CE')


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a
    vector that represents the probability of the sample belonging to each
    cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=3))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)`
        witch represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution.
        Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(
            shape=(self.n_clusters, input_dim),
            initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
         Measure the similarity between embedded point z_i and centroid µ_j.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning
                    sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample.
            shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0
            + (K.sum(K.square(K.expand_dims(inputs, axis=1)
            - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        # Make sure each sample's 3 values add up to 1.
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# ------------------------------------------------------------------------------

def CAE_Conv2DTranspose_OLD(input_shape=(96, 96, 1), filters=[8, 16, 32, 64, 128, 256, 512, 1024, 30]):

    input_img = Input(shape=input_shape)

    # Encoder
    x = Conv2D(filters[0], 5, strides=2, padding='same', activation='relu',
               name='conv1', input_shape=input_shape)(input_img)
    x = Conv2D(filters[1], 3, strides=2, padding='same',
               activation='relu', name='conv2')(x)
    x = Conv2D(filters[2], 3, strides=2, padding='same',
               activation='relu', name='conv3')(x)
    x = Conv2D(filters[3], 3, strides=2, padding='same',
               activation='relu', name='conv4')(x)
    x = Conv2D(filters[4], 3, strides=2, padding='same',
               activation='relu', name='conv5')(x)
    x = Conv2D(filters[5], 3, strides=2, padding='same',
               activation='relu', name='conv6')(x)

    x = Flatten()(x)

    encoded = Dense(units=filters[-1], name='embedding')(x)

    y = Dense(units=3, name='input_clustering')(encoded)

    # Decoder
    x = Dense(units=filters[5]*int(input_shape[0]/64) *
              int(input_shape[0]/64), activation='relu')(encoded)

    # Reshape into an image of the same shape as before our last `Flatten` layer
    x = Reshape((int(input_shape[0]/64),
                 int(input_shape[0]/64), filters[5]))(x)

    x = Conv2DTranspose(
        filters[4], 3, strides=2, padding='same', activation='relu', name='deconv6')(x)
    x = Conv2DTranspose(
        filters[3], 3, strides=2, padding='same', activation='relu', name='deconv5')(x)
    x = Conv2DTranspose(
        filters[2], 3, strides=2, padding='same', activation='relu', name='deconv4')(x)
    x = Conv2DTranspose(
        filters[1], 3, strides=2, padding='same', activation='relu', name='deconv3')(x)
    x = Conv2DTranspose(
        filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2')(x)

    decoded = Conv2DTranspose(
        input_shape[2], 5, strides=2, padding='valid', name='deconv1')(x)

    return Model(inputs=input_img, outputs=decoded, name='CAE_Conv2DTranspose'), Model(inputs=input_img, outputs=[encoded, y], name='CE')

# ----------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import pdb; pdb.set_trace()

from keras.layers import Input, Dense, Conv2D, Flatten, Reshape, Conv2DTranspose
from keras.models import Model


def CAE_Conv2DTranspose(input_shape=(96, 96, 1), filters=[8, 16, 32, 64, 128, 256, 512, 1024, 30]):

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

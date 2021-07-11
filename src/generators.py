from tensorflow.keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator
import numpy as np


def generator(image_generator, x, y=None, sample_weight=None, batch_size=16, shuffle=True):
    """
    Data generator that supplies training batches for Model().fit_generator.
    :param image_generator: MyImageGenerator, defines and applies transformations for the input images
    :param x: input image data, supports shape=[n_samples, width, height, channels] and [n_samples, n_features]
    :param y: the target of the network's output
    :param sample_weight: weight for x, shape=[n_samples]
    :param batch_size: batch size
    :param shuffle: whether to shuffle the data
    :return: An iterator, outputs [batch_x, batch_x] or [batch_x, batch_y] or
            [batch_x, batch_y, batch_sample_weight] each time
    """
    if len(x.shape) > 2:  # image
        gen0_idx = image_generator.flow(x, shuffle=shuffle, batch_size=batch_size)
        while True:
            batch_x, idx = gen0_idx.next()
            result = [batch_x] + \
                     [batch_x if y is None else y[idx]] + \
                     ([] if sample_weight is None else [sample_weight[idx]])
            yield tuple(result)
    else:  # if the sample is represented by vector, need to reshape to matrix and then flatten back
        width = int(np.sqrt(x.shape[-1]))
        if width * width == x.shape[-1]:  # gray
            im_shape = [-1, width, width, 1]
        else:  # RGB
            width = int(np.sqrt(x.shape[-1] / 3.0))
            im_shape = [-1, width, width, 3]
        gen0 = image_generator.flow(np.reshape(x, im_shape), shuffle=shuffle, batch_size=batch_size)
        while True:
            batch_x, idx = gen0.next()
            batch_x = np.reshape(batch_x, [batch_x.shape[0], x.shape[-1]])
            result = [batch_x] + \
                     [batch_x if y is None else y[idx]] + \
                     ([] if sample_weight is None else [sample_weight[idx]])
            yield tuple(result)


class MyIterator(NumpyArrayIterator):
    """
    The only difference with NumpyArrayIterator is this.next() returns (samples, index) while NumpyArrayIterator
    returns samples
    """
    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array), index_array


class MyImageGenerator(ImageDataGenerator):
    """
    The only difference with ImageDataGenerator is this.flow().next() returns (samples, index) while ImageDataGenerator
    returns samples
    """
    def flow(self, x, y=None, batch_size=16, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='png'):
        return MyIterator(
            x, y, self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)


def generators(x_train, x_val, batch_size):
    # define data augmentation configuration
    train_datagen = MyImageGenerator(
        rescale=1./225,
        # featurewise_center=True,
        # featurewise_std_normalization=True,
    )
    val_datagen = MyImageGenerator(
        rescale=1./225,
        # featurewise_center=True,
        # featurewise_std_normalization=True,
    )

    # fit the data augmentation
    train_datagen.fit(x_train)
    val_datagen.fit(x_val)

    # setup generator
    train_generator = train_datagen.flow(
        x_train,
        x_train,
        batch_size=batch_size,
    )
    val_generator = val_datagen.flow(
        x_val,
        x_val,
        batch_size=batch_size,
        shuffle=False
    )

    return train_generator, val_generator

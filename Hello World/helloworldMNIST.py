# Loading MNIST dataset in Keras
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# The network architecture
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,))) # This layers are densely connected between eachother
network.add(layers.Dense(10, activation='softmax'))


# The compilation step
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Preparing the image data
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# Preparing the test_labels
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Train the network via a call to the network's fit method
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# Check performance
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

# Result: We quickly reach an accuracy of 0.989 on the training data.
#   The test-set accuracy turns out to be 97.8%, quite a bit lower than
#   the training data: test_acc: 0.9773. This gap between training accuracy
#   and test accuracy is an example of overfitting.

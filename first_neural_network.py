from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images.shape
len(train_labels)
print(train_labels)

from keras import models
from keras import layers

network = models.sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

'''compile this network'''
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
'''train images are of type uint8 with values in [0,255]'''
train_images = train_images.reshape('float32')/255

train_images = train_images.reshape((10000, 28 * 28))
train_images = train_images.reshape('float32')/255

from keras.util import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

print(train_labels[0])

'''now we will fit the data into the network 
data is sent through the data 5 times in a batch size of 128'''
network.fit(train_images, train_labels, epochs=5, batch_size=128)

'''evaluating the network on test data set'''
test_loss, test_acc = network.evaluate(test_images, test_labels)

print('test_acc:', test_acc)
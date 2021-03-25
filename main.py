import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def get_datasets(name):
    if name == "fashion_mnist":
        #load in the fashion dataset which is 60000 train and 10000 test
        (train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
        train_images = train_images.reshape(60000,28,28,1)
        test_images = test_images.reshape(10000,28,28,1)
        
        # Split training data into training and validation data
        (train_images, train_labels) = train_images[5000:], train_labels[5000:]
        (valid_images, valid_labels) = train_images[:5000], train_labels[:5000]

        # Normalise colour data
        train_images = train_images / 255
        valid_images = valid_images / 255
        test_images = test_images / 255
        return [(train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels)]
    print("Dataset was not found")
    exit(1)

def init_model_example(verbose=0):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    if verbose == 1:
        model.summary()
    return model


def init_model_1(verbose=0):
    model = models.Sequential()
    model.add(layers.Conv2D(filters=8, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
    model.add(layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
    model.add(layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    if verbose == 1:
        model.summary()
    return model


def init_model_2(verbose=0):
    model = models.Sequential()
    model.add(layers.Conv2D(filters=8, kernel_size=(5,5), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(filters=64, kernel_size=(5,5), activation='relu'))
    model.add(layers.Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
    model.add(layers.Conv2D(filters=16, kernel_size=(5,5), activation='relu'))
    model.add(layers.Conv2D(filters=128, kernel_size=(5,5), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    if verbose == 1:
        model.summary()
    return model

def init_model_3(verbose=0):
    model = models.Sequential()
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    if verbose == 1:
        model.summary()
    return model

def init_model_4(verbose=0):
    model = models.Sequential()
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), input_shape=(28, 28, 1)))
    model.add(layers.LeakyReLU(alpha=0.3))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3)))
    model.add(layers.LeakyReLU(alpha=0.3))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3)))
    model.add(layers.LeakyReLU(alpha=0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    if verbose == 1:
        model.summary()
    return model

def init_model_5(verbose=0):
    model = models.Sequential()
    model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    if verbose == 1:
        model.summary()
    return model


def plot_training_loss(model, history, test_images, test_labels):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print("test loss = ", test_loss)
    print("test accuracy = ", test_acc)
    plt.show()


def main():
    data = get_datasets("fashion_mnist")
    (train_images, train_labels) = data[0]
    (valid_images, valid_labels) = data[1]
    (test_images, test_labels) = data[2]

    # model = init_model_example()
    # model = init_model_1()
    model = init_model_1()
    # model = init_model_4()
    # model = init_model_5()

    # We moeten eerst alleen evaluaten met de validation data en niet de test data.
    # dus het is niet de bedoeling dat we model evaluate met test data doen toch?
    history = model.fit(train_images,
                    train_labels,
                    batch_size=64,
                    epochs=15,
                    verbose=2,
                    validation_data=(valid_images, valid_labels))

    plot_training_loss(model, history, test_images, test_labels)

    # model.save('Models\\model_3')
    # model.save('Models\\model_4')
    # model.save('Models\\model_5')
    # model = tf.keras.models.load_model('Models/model_example')


    # output = model.predict(test_images[:4])
    # print(output.argmax(axis=-1))
    # print(test_labels[:4])

if __name__ == "__main__":
    # physical_devices = tf.config.list_physical_devices('GPU')
    # print(physical_devices)
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    main()
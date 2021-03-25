import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def get_datasets(name):
    if name == "fashion_mnist_training":
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
    elif name == "fashion_mnist_testing":
        # load in the fashion dataset which is 60000 train and 10000 test
        (train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
        train_images = train_images.reshape(60000, 28, 28, 1)
        test_images = test_images.reshape(10000, 28, 28, 1)

        # Normalise colour data
        train_images = train_images / 255
        test_images = test_images / 255
        return [(train_images, train_labels), (test_images, test_labels)]

    print("Dataset was not found")
    exit(1)

def init_model_1(verbose=0):
    model = models.Sequential()
    model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
    model.add(layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu'))

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
    model.add(layers.Conv2D(filters=64, kernel_size=(5,5), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
    model.add(layers.Conv2D(filters=16, kernel_size=(5,5), activation='relu'))

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
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    if verbose == 1:
        model.summary()
    return model

def training_stage():
    data = get_datasets("fashion_mnist_training")
    (train_images, train_labels) = data[0]
    (valid_images, valid_labels) = data[1]
    # (test_images, test_labels) = data[2]

    model = init_model_1()
    history = get_history(model, valid_images, valid_labels, train_images, train_labels)
    plot_training_loss(history)
    model.save('Training_Models\\model_1')

    model = init_model_2()
    history = get_history(model, valid_images, valid_labels, train_images, train_labels)
    plot_training_loss(history)
    model.save('Training_Models\\model_2')

    model = init_model_3()
    history = get_history(model, valid_images, valid_labels, train_images, train_labels)
    plot_training_loss(history)
    model.save('Training_Models\\model_3')

    model = init_model_4()
    history = get_history(model, valid_images, valid_labels, train_images, train_labels)
    plot_training_loss(history)
    model.save('Training_Models\\model_4')

    model = init_model_5()
    history = get_history(model, valid_images, valid_labels, train_images, train_labels)
    plot_training_loss(history)
    model.save('Training_Models\\model_5')

def testing_stage():
    i = 0

def decay_rate(epoch, lr):
    decay_step = 5
    if epoch % decay_step == 0 and not epoch == 0:
        return lr / 2
    else:
        return lr


def get_history(model, valid_images, valid_labels, train_images, train_labels):
    callback = tf.keras.callbacks.LearningRateScheduler(decay_rate, verbose=1)
    history = model.fit(train_images,
                        train_labels,
                        batch_size=64,
                        epochs=15,
                        callbacks=[callback],
                        verbose=2,
                        validation_data=(valid_images, valid_labels))
    return history

def plot_training_loss(history):
    plt.plot(np.log(history.history['loss']), label='training')
    plt.plot(np.log(history.history['val_loss']), label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.ylim([0.4, 0.7])
    plt.legend(loc='upper right')
    print(history.history)
    plt.show()

def evalueate_test_data(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print("test loss = ", test_loss)
    print("test accuracy = ", test_acc)

def main():
    training_stage()
    # model = tf.keras.models.load_model('Models\\model_3')


    # output = model.predict(test_images[:4])
    # print(output.argmax(axis=-1))
    # print(test_labels[:4])

if __name__ == "__main__":
    # physical_devices = tf.config.list_physical_devices('GPU')
    # print(physical_devices)
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    main()
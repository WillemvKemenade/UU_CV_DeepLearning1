import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models


def get_datasets(name):
    if name == "fashion_mnist":
        #load in the fashion dataset which is 60000 train and 10000 test
        (train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
        
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


def plot_training_loss(model, history, test_images, test_labels):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
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
    print(test_images.shape)
    print(train_images.shape)
    print(valid_images.shape)

    # model = init_model_example()

    # history = model.fit(train_images, 
    #                 train_labels,
    #                 epochs=10,
    #                 validation_data=(valid_images, valid_labels),
    #                 verbose=2)

    # plot_training_loss(model, history, test_images, test_labels)

    # model.save('Models/model_example')
    model = tf.keras.models.load_model('Models/model_example')


    output = model.predict(test_images[:4])
    print(output.argmax(axis=-1))
    print(test_labels[:4])

if __name__ == "__main__":
    # physical_devices = tf.config.list_physical_devices('GPU')
    # print(physical_devices)
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    main()
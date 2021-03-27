import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import KFold
import os
import functools
import keras
import pandas as pd
import seaborn as sns

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

def init_model_1(verbose=0, training=True):
    model = models.Sequential()
    model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
    model.add(layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    if training == True:
        top1_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=1)
        top1_acc.__name__ = 'top1_acc'
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy', 'top_k_categorical_accuracy', top1_acc])
    else:
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

    top1_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=1)
    top1_acc.__name__ = 'top1_acc'
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy', 'top_k_categorical_accuracy', top1_acc])
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

    top1_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=1)
    top1_acc.__name__ = 'top1_acc'
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy', 'top_k_categorical_accuracy', top1_acc])
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

    top1_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=1)
    top1_acc.__name__ = 'top1_acc'
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy', 'top_k_categorical_accuracy', top1_acc])
    if verbose == 1:
        model.summary()
    return model

def init_model_5(verbose=0, training=True):
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

    if training == True:
        top1_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=1)
        top1_acc.__name__ = 'top1_acc'
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy', 'top_k_categorical_accuracy', top1_acc])
    else:
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

    model = init_model_1(0, True)
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

    model = init_model_5(0, True)
    history = get_history(model, valid_images, valid_labels, train_images, train_labels)
    plot_training_loss(history)
    model.save('Training_Models\\model_5')

def KFold_Stage():
    data = get_datasets("fashion_mnist_training")
    (train_images, train_labels) = data[0]

    model = init_model_1(0, True)
    LOSS, VAL_LOSS = KFold_Model(model, 1, train_images, train_labels)
    AVERAGE_LOSS, AVERAGE_VAL_LOSS = Get_Average_Loss(LOSS, VAL_LOSS)
    plot_training_loss(AVERAGE_LOSS, AVERAGE_VAL_LOSS, 1)

    model = init_model_2()
    LOSS, VAL_LOSS = KFold_Model(model, 2, train_images, train_labels)
    AVERAGE_LOSS, AVERAGE_VAL_LOSS = Get_Average_Loss(LOSS, VAL_LOSS)
    plot_training_loss(AVERAGE_LOSS, AVERAGE_VAL_LOSS, 2)

    model = init_model_3()
    LOSS, VAL_LOSS = KFold_Model(model, 3, train_images, train_labels)
    AVERAGE_LOSS, AVERAGE_VAL_LOSS = Get_Average_Loss(LOSS, VAL_LOSS)
    plot_training_loss(AVERAGE_LOSS, AVERAGE_VAL_LOSS, 3)

    model = init_model_4()
    LOSS, VAL_LOSS = KFold_Model(model, 4, train_images, train_labels)
    AVERAGE_LOSS, AVERAGE_VAL_LOSS = Get_Average_Loss(LOSS, VAL_LOSS)
    plot_training_loss(AVERAGE_LOSS, AVERAGE_VAL_LOSS, 4)

    model = init_model_5(0, True)
    LOSS, VAL_LOSS = KFold_Model(model, 5, train_images, train_labels)
    AVERAGE_LOSS, AVERAGE_VAL_LOSS = Get_Average_Loss(LOSS, VAL_LOSS)
    plot_training_loss(AVERAGE_LOSS, AVERAGE_VAL_LOSS, 5)

def testing_stage():
    data = get_datasets("fashion_mnist_testing")
    (train_images, train_labels) = data[0]
    (test_images, test_labels) = data[1]

    model = init_model_1(0, False)
    history = get_history(model, test_images, test_labels, train_images, train_labels)
    plot_testing_loss(history)
    evaluate_test_data(model, test_images, test_labels)
    confusion_matrix(model, test_images, test_labels)
    model.save('Test_Models\\model_1')

    model = init_model_5(0, False)
    history = get_history(model, test_images, test_labels, train_images, train_labels)
    plot_testing_loss(history)
    evaluate_test_data(model, test_images, test_labels)
    model.save('Test_Models\\model_5')

def confusion_matrix(model, test_images, test_labels):
    y_pred = model.predict_classes(test_images)
    confusion_matrix_model = tf.math.confusion_matrix(labels=test_labels, predictions=y_pred).numpy()
    confusion_normalization = np.around(confusion_matrix_model.astype('float') / confusion_matrix_model.sum(axis=1)[:, np.newaxis], decimals=2)

    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    confusion_dataframe = pd.DataFrame(confusion_normalization, index=classes, columns=classes)

    plt.figure(figsize=(10, 10))
    sns.heatmap(confusion_dataframe, annot=True, cmap=plt.cm.Blues)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def KFold_Model(model, model_number, train_images, train_labels):
    folds = 5
    kfold = KFold(n_splits=folds)
    LOSS = []
    VAL_LOSS = []
    print("MODEL "+str(model_number)+" ===================================================================================================")
    for train, validation in kfold.split(train_images):
        print('train: %s, test: %s' % (train, validation))
        train_images_cross, train_labels_cross = train_images[train], train_labels[train]
        val_images_cross, val_labels_cross = train_images[validation], train_labels[validation]
        history = get_history(model, val_images_cross, val_labels_cross, train_images_cross, train_labels_cross)

        LOSS.append(history.history['loss'])
        VAL_LOSS.append(history.history['val_loss'])
    model.save('KFold_Models\\model_'+str(model_number))
    return LOSS, VAL_LOSS

def Get_Average_Loss(LOSS, VAL_LOSS):
    AVERAGE_LOSS = []
    AVERAGE_VAL_LOSS = []
    for x in range(15):
        fold_1 = LOSS[0][x]
        fold_2 = LOSS[1][x]
        fold_3 = LOSS[2][x]
        fold_4 = LOSS[3][x]
        fold_5 = LOSS[4][x]
        average_loss = (fold_1 + fold_2 + fold_3 + fold_4 + fold_5) / 5
        AVERAGE_LOSS.append(average_loss)

        fold_val_1 = VAL_LOSS[0][x]
        fold_val_2 = VAL_LOSS[1][x]
        fold_val_3 = VAL_LOSS[2][x]
        fold_val_4 = VAL_LOSS[3][x]
        fold_val_5 = VAL_LOSS[4][x]
        average_val_loss = (fold_val_1 + fold_val_2 + fold_val_3 + fold_val_4 + fold_val_5) / 5
        AVERAGE_VAL_LOSS.append(average_val_loss)
    return AVERAGE_LOSS, AVERAGE_VAL_LOSS

def decay_rate(epoch, lr):
    decay_step = 5
    if epoch % decay_step == 0 and not epoch == 0:
        return lr / 2
    elif epoch == 0:
        lr = 0.0010000000474974513
        return lr
    else:
        return lr


def get_history(model, valid_test_images, valid_test_labels, train_images, train_labels):
    callback = tf.keras.callbacks.LearningRateScheduler(decay_rate, verbose=1)
    history = model.fit(train_images,
                        train_labels,
                        batch_size=64,
                        epochs=15,
                        callbacks=[callback],
                        verbose=2,
                        validation_data=(valid_test_images, valid_test_labels))
    return history

def plot_training_loss(AVERAGE_LOSS, AVERAGE_VAL_LOSS, number):
    plt.plot(np.log(AVERAGE_LOSS), label='training')
    plt.plot(np.log(AVERAGE_VAL_LOSS), label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.ylim([0.4, 0.6])
    plt.title("Loss Model "+str(number))
    plt.legend(loc='upper right')
    plt.show()

def plot_testing_loss(history):
    plt.plot(np.log(history.history['loss']), label='training')
    plt.plot(np.log(history.history['val_loss']), label='testing')
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.ylim([0.4, 0.7])
    plt.legend(loc='upper right')
    plt.show()

def evaluate_test_data(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print("test loss = ", test_loss)
    print("test accuracy = ", test_acc)

def main():
    training_stage()
    testing_stage()
    KFold_Stage()

if __name__ == "__main__":
    main()
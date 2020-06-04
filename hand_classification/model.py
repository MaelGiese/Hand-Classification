import tensorflow.keras as keras
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization

import hand_classification.utils.build_dataset_from_directory as dataset
import hand_classification.utils.data_augmentation as data_augmentation
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def train():
    batch_size = 512
    epochs = 50
    learning_rate = 0.01
    model_name = "models/hand_classification_model.h5"

    train_path = 'Dataset/Train/'
    test_path = 'Dataset/Test/'
    IMAGE_WIDTH = 64
    IMAGE_HEIGHT = 64
    input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)

    augment_data = True
    merge_train_and_test = False

    # Build train dataset
    x_train, y_train = dataset.build_dataset(train_path, IMAGE_WIDTH, IMAGE_HEIGHT)
    # Build test dataset
    x_test, y_test = dataset.build_dataset(test_path, IMAGE_WIDTH, IMAGE_HEIGHT)

    num_classes = len(np.unique(y_train))

    ############################ Add augmented data to the training data ####################################
    if augment_data:
        x_train_augmented, y_train_augmented = data_augmentation.build_augmented_dataset(train_path,
                                                                                         IMAGE_WIDTH, IMAGE_HEIGHT,
                                                                                         flipped=True, saturated=True,
                                                                                         bright=True, cropped=True,
                                                                                         grayscaled=True)
        x_train = np.concatenate((x_train, x_train_augmented), axis=0)
        y_train = y_train + y_train_augmented

    ############################# Add test dataset to the train dataset ########################################
    if merge_train_and_test:
        x_test_augmented, y_test_augmented = data_augmentation.build_augmented_dataset(test_path,
                                                                                         IMAGE_WIDTH, IMAGE_HEIGHT,
                                                                                         flipped=True, saturated=True,
                                                                                         bright=True, cropped=True,
                                                                                         grayscaled=True)
        x_test = np.concatenate((x_test, x_test_augmented), axis=0)
        y_test = y_test + y_test_augmented

        x_train = np.concatenate((x_train, x_test), axis=0)
        y_train = y_train + y_test

    # Shuffle the training data
    x_train, y_train = shuffle(x_train, y_train)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    ####### Model structure #######
    model = Sequential()

    model.add(Conv2D(16, kernel_size=(7, 7), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())



    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=learning_rate),
                  metrics=['acc'])

    ####### TRAINING #######
    hist = model.fit(x_train, y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=2,
                     validation_data=(x_test, y_test))

    # Evaluation
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # plotting the metrics
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')

    plt.subplot(2, 1, 2)
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')

    plt.tight_layout()
    plt.show()

    model.save(model_name)


if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    train()

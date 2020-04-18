
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import hand_classification.utils.build_dataset_from_directory as dataset
import numpy as np
import os
import matplotlib.image as mpimg


def create_augmented_data_from_directory(path, batch_size, IMG_HEIGHT, IMG_WIDTH):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        class_mode='binary')
    print(train_generator)




def data_augmentation(x_train, y_train, x_test, y_test):
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        brightness_range=(0.1, 0.9),
        rotation_range=20,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        horizontal_flip=True)

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(x_train)
    augmented_data = datagen.flow(x_train, y_train, batch_size=32)
    print(augmented_data)
    """
    # fits the model on batches with real-time data augmentation:
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                        steps_per_epoch=len(x_train) / 32,
                        epochs=epochs,
                        validation_data=(x_test, y_test))

    return model
    """



def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    path = "Dataset/"
    IMAGE_WIDTH = 64
    IMAGE_HEIGHT = 64
    batch_size = 256

    x_train, x_test, y_train, y_test = dataset.load_data(path, IMAGE_WIDTH, IMAGE_HEIGHT)

    data_augmentation(x_train, y_train, x_test, y_test)

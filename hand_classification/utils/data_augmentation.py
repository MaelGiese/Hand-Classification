from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os


def build_augmented_dataset(dir_path, IMAGE_WIDTH, IMAGE_HEIGHT, flipped=False, saturated=False, bright=False,
                            cropped=False, grayscaled=False):
    images = []
    images_class = []
    count_classes = 0
    valid_images = [".jpg", ".png"]

    hand_classes = os.listdir(dir_path)

    for hand_class in hand_classes:
            print(">> Working on class : " + hand_class)

            sets = os.listdir(dir_path + hand_class)
            for set in sets:
                image_dir = dir_path + hand_class + '/' + set + '/'

                for f in os.listdir(image_dir):
                    ext = os.path.splitext(f)[1]
                    if ext.lower() in valid_images:
                        image_string = tf.io.read_file(os.path.join(image_dir, f))
                        image = tf.image.decode_png(image_string, channels=3)
                        image = tf.image.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

                        if flipped:
                            flipped_img = tf.image.flip_left_right(image)
                            flipped_img = tf.keras.preprocessing.image.img_to_array(flipped_img)
                            images.append(flipped_img)
                            images_class.append(count_classes)

                        if saturated:
                            saturated_img = tf.image.adjust_saturation(image, 3)
                            saturated_img = tf.keras.preprocessing.image.img_to_array(saturated_img)
                            images.append(saturated_img)
                            images_class.append(count_classes)

                        if bright:
                            bright_img = tf.image.adjust_brightness(image, 0.4)
                            bright_img = tf.keras.preprocessing.image.img_to_array(bright_img)
                            images.append(bright_img)
                            images_class.append(count_classes)

                        if cropped:
                            cropped_img = tf.image.central_crop(image, central_fraction=0.4)
                            cropped_img = tf.image.resize(cropped_img, (IMAGE_WIDTH, IMAGE_HEIGHT))
                            cropped_img = tf.keras.preprocessing.image.img_to_array(cropped_img)

                            images.append(cropped_img)
                            images_class.append(count_classes)

                        if grayscaled:
                            grayscaled_img = tf.image.rgb_to_grayscale(image)
                            grayscaled_img = tf.image.grayscale_to_rgb(grayscaled_img)
                            grayscaled_img = tf.keras.preprocessing.image.img_to_array(grayscaled_img)
                            images.append(grayscaled_img)
                            images_class.append(count_classes)

            count_classes += 1

    images = np.array(images)
    images = (images / 127.5) - 1

    return images, images_class


def create_augmented_data_from_directory(path, IMG_WIDTH, IMG_HEIGHT):
    image_string = tf.io.read_file(path)
    image = tf.image.decode_png(image_string, channels=3)

    saturated = tf.image.adjust_saturation(image, 3)
    visualize(image, saturated)

    bright = tf.image.adjust_brightness(image, 0.4)
    visualize(image, bright)


def visualize(original, augmented):
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Original image')
    plt.imshow(original)

    plt.subplot(1, 2, 2)
    plt.title('Augmented image')
    plt.imshow(augmented)

    plt.show()


if __name__ == "__main__":
    dir_path = '../Dataset/Train/'
    IMAGE_WIDTH = 28
    IMAGE_HEIGHT = 28

    x, y = build_augmented_dataset(dir_path, IMAGE_WIDTH, IMAGE_HEIGHT, flipped=False, saturated=False, bright=False,
                                   cropped=True, grayscaled=False)

    print(x.shape)
    print(y)

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
                            # Flip rigth-left
                            flipped_img = tf.image.flip_left_right(image)
                            flipped_img = tf.keras.preprocessing.image.img_to_array(flipped_img)
                            images.append(flipped_img)
                            images_class.append(count_classes)

                            # Flip up-down
                            flipped_img = tf.image.flip_up_down(image)
                            flipped_img = tf.keras.preprocessing.image.img_to_array(flipped_img)
                            images.append(flipped_img)
                            images_class.append(count_classes)


                        if grayscaled:
                            grayscaled_img = tf.image.rgb_to_grayscale(image)
                            grayscaled_img = tf.image.grayscale_to_rgb(grayscaled_img)
                            grayscaled_img = tf.keras.preprocessing.image.img_to_array(grayscaled_img)
                            images.append(grayscaled_img)
                            images_class.append(count_classes)

                        if saturated:
                            # 3
                            saturated_img = tf.image.adjust_saturation(image, 3)
                            saturated_img = tf.keras.preprocessing.image.img_to_array(saturated_img)
                            images.append(saturated_img)
                            images_class.append(count_classes)

                            # 5
                            saturated_img = tf.image.adjust_saturation(image, 5)
                            saturated_img = tf.keras.preprocessing.image.img_to_array(saturated_img)
                            images.append(saturated_img)
                            images_class.append(count_classes)

                        if bright:
                            # 0.2
                            bright_img = tf.image.adjust_brightness(image, 0.2)
                            bright_img = tf.keras.preprocessing.image.img_to_array(bright_img)
                            images.append(bright_img)
                            images_class.append(count_classes)

                            # 0.4
                            bright_img = tf.image.adjust_brightness(image, 0.4)
                            bright_img = tf.keras.preprocessing.image.img_to_array(bright_img)
                            images.append(bright_img)
                            images_class.append(count_classes)

                            # 0.6
                            bright_img = tf.image.adjust_brightness(image, 0.6)
                            bright_img = tf.keras.preprocessing.image.img_to_array(bright_img)
                            images.append(bright_img)
                            images_class.append(count_classes)

                        if cropped:
                            # 0.2
                            cropped_img = tf.image.central_crop(image, central_fraction=0.2)
                            cropped_img = tf.image.resize(cropped_img, (IMAGE_WIDTH, IMAGE_HEIGHT))
                            cropped_img = tf.keras.preprocessing.image.img_to_array(cropped_img)
                            images.append(cropped_img)
                            images_class.append(count_classes)

                            # 0.4
                            cropped_img = tf.image.central_crop(image, central_fraction=0.4)
                            cropped_img = tf.image.resize(cropped_img, (IMAGE_WIDTH, IMAGE_HEIGHT))
                            cropped_img = tf.keras.preprocessing.image.img_to_array(cropped_img)
                            images.append(cropped_img)
                            images_class.append(count_classes)

            count_classes += 1

    images = np.array(images)
    images = (images / 127.5) - 1

    return images, images_class


def visualize(original, augmented, augmented2):
    fig = plt.figure()
    plt.subplot(1, 3, 1)
    plt.title('Original image')
    plt.imshow(original)

    plt.subplot(1, 3, 2)
    plt.title('Flipped image')
    plt.imshow(augmented)

    plt.subplot(1, 3, 3)
    plt.title('Saturated image')
    plt.imshow(augmented2)

    plt.show()


if __name__ == "__main__":
    dir_path = '../Dataset/Test/'
    IMAGE_WIDTH = 64
    IMAGE_HEIGHT = 64

    x, y = build_augmented_dataset(dir_path, IMAGE_WIDTH, IMAGE_HEIGHT, flipped=False, saturated=False, bright=False,
                                  cropped=True, grayscaled=False)
    print(x.shape)
    print(y)

    """
    image_string = tf.io.read_file(os.path.join(dir_path, 'Palm/SET_0/Palm_104.png'))
    image = tf.image.decode_png(image_string, channels=3)
    image = tf.image.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

    flipped_img = tf.image.flip_left_right(image)
    flipped_img = tf.keras.preprocessing.image.img_to_array(flipped_img)

    saturated_img = tf.image.adjust_saturation(image, 5)
    saturated_img = tf.keras.preprocessing.image.img_to_array(saturated_img)

    image = tf.keras.preprocessing.image.img_to_array(image)

    image = (image / 127.5)
    flipped_img = (flipped_img / 127.5)
    saturated_img = (saturated_img / 127.5)

    visualize(image, flipped_img, saturated_img)
    """


import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf


def build_dataset(dir_path, IMAGE_WIDTH, IMAGE_HEIGHT):
    images = []
    images_class = []
    count_classes = 0
    valid_images = [".jpg", ".png"]

    hand_classes = os.listdir(dir_path)
    req_poses = hand_classes.copy()

    for hand_class in hand_classes:
        if hand_class in req_poses:
            print(">> Working on class : " + hand_class)

            sets = os.listdir(dir_path + hand_class)
            for set in sets:
                image_dir = dir_path + hand_class + '/' + set + '/'

                for f in os.listdir(image_dir):
                    ext = os.path.splitext(f)[1]
                    if ext.lower() in valid_images:
                        image = cv2.imread(os.path.join(image_dir, f))
                        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
                        images.append(image)
                        images_class.append(count_classes)
            count_classes += 1

    images = np.array(images)
    images = (images / 127.5) - 1

    return images, images_class


if __name__ == '__main__':
    dir_path = '../Dataset/Train/'
    IMAGE_WIDTH = 28
    IMAGE_HEIGHT = 28

    x, y = build_dataset(dir_path, IMAGE_WIDTH, IMAGE_HEIGHT)

    print(x.shape)
    print(y)

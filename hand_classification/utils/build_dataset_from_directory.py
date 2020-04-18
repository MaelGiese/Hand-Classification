import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


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

            for f in os.listdir(dir_path + hand_class):
                ext = os.path.splitext(f)[1]
                if ext.lower() in valid_images:
                    image = cv2.imread(os.path.join(dir_path + hand_class, f))
                    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
                    images.append(image)
                    images_class.append(count_classes)
            count_classes += 1

    images = np.array(images)
    images = (images / 127.5) - 1

    return images, images_class


def load_data(dir_path, IMAGE_WIDTH, IMAGE_HEIGHT):
    x, y = build_dataset(dir_path, IMAGE_WIDTH, IMAGE_HEIGHT)
    x, y = shuffle(x, y, random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/4)

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    dir_path = '../Dataset/'
    IMAGE_WIDTH = 28
    IMAGE_HEIGHT = 28

    x_train, x_test, y_train, y_test = load_data(dir_path, IMAGE_WIDTH, IMAGE_HEIGHT)

    print(x_train.shape)
    print(y_train)
    print(x_test.shape)
    print(y_test)

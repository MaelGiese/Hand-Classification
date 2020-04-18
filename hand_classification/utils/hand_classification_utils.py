import os

# Use the CPU not the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow import Graph
import cv2


def load_KerasGraph(path):
    print("> ====== loading Keras model for classification")
    thread_graph = Graph()
    with thread_graph.as_default():
        thread_session = tf.compat.v1.Session()
        with thread_session.as_default():
            model = keras.models.load_model(path)
            graph = tf.compat.v1.get_default_graph()
    print(">  ====== Keras model loaded")
    return model, graph, thread_session


def classify(model, graph, sess, img, IMAGE_WIDTH, IMAGE_HEIGHT):
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))

    # Convert to float values between -1 and 1
    img = img.astype(dtype="float64")
    img = (img / 127.5) - 1
    res = np.reshape(img, (1, IMAGE_WIDTH, IMAGE_HEIGHT, 3))

    with graph.as_default():
        with sess.as_default():
            prediction = model.predict(res)

    return prediction[0]


def test_classify(model, img, IMAGE_WIDTH, IMAGE_HEIGHT):
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))

    # Convert to float values between -1 and 1
    img = img.astype(dtype="float64")
    img = (img / 127.5) - 1
    res = np.reshape(img, (1, IMAGE_WIDTH, IMAGE_HEIGHT, 3))

    prediction = model.predict(res)

    return prediction[0]


if __name__ == "__main__":
    classification_graph_path = '../models/hand_classification_model.h5'
    IMAGE_WIDTH = 64
    IMAGE_HEIGHT = 64

    print(">> loading keras model for pose classification")

    model = keras.models.load_model(classification_graph_path)

    # Fist
    print('<< FIST >>')
    img = cv2.imread("../Dataset/Fist/Fist_0.png")
    print(test_classify(model, img, IMAGE_WIDTH, IMAGE_HEIGHT))

    # Palm
    print('<< PALM >>')
    img = cv2.imread("../Dataset/Palm/Palm_0.png")
    print(test_classify(model, img, IMAGE_WIDTH, IMAGE_HEIGHT))

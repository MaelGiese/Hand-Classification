import threading
from threading import Thread

from multiprocessing import Queue

import cv2

import dino_game.game as game
import hand_classification_multi_threaded_v2 as classifier

import sys
import os


def main():
    jump_q = Queue(maxsize=1)
    cv2.namedWindow('Multi-Threaded Detection', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Cropped', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Classes probalities', cv2.WINDOW_NORMAL)

    classifier_thread = Thread(target=classifier.main, args=(jump_q,))
    classifier_thread.start()

    game.main(jump_q)


if __name__ == '__main__':
    main()
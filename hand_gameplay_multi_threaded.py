from threading import Thread
from multiprocessing import Queue
import dino_game.game as game
import hand_classification_multi_threaded as classifier

import sys
import os


def main():
    jump_q = Queue(maxsize=1)

    classifier_thread = Thread(target=classifier.main, args=(jump_q,))
    game_thread = Thread(target=game.main, args=(jump_q,))

    classifier_thread.start()
    game_thread.start()


if __name__ == '__main__':
    main()
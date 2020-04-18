from threading import Thread
from multiprocessing import Queue
import dino_game.game as game
import hand_classification_multi_threaded as classifier

jump = False


def game_worker(jump):
    game.main(jump)


def start_detection(jump):
    classifier.main(jump)


def main():
    jump_q = Queue(maxsize=1)

    classifier = Thread(target=start_detection, args=(jump_q,))
    game = Thread(target=game_worker, args=(jump_q,))

    classifier.start()
    game.start()


if __name__ == '__main__':
    main()

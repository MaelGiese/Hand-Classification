import cv2
import numpy as np
from random import randint


def draw_classes_probabilities(values, width, height, names=None):
    if names is None:
        names = ['', '', '', '', '', '']

    nb_classes = len(values)
    left_margin = 150
    margin = 50
    thickness = 40

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    blank = np.zeros((height, width, 3), np.uint8)

    for i in range(nb_classes):
        if values[i] > 0.7:
            cv2.rectangle(blank, (left_margin, margin + int(margin * i)),
                          (left_margin + int(values[i] * 200), margin + thickness + int(margin * i)), (0, 255, 0), -1)
        else:
            cv2.rectangle(blank, (left_margin, margin + int(margin * i)),
                          (left_margin + int(values[i] * 200), margin + thickness + int(margin * i)), (255, 0, 0), -1)
        cv2.putText(blank, names[i], (0, margin + int(margin * i) + int(thickness / 2)), font, fontScale, fontColor,
                    lineType)
        cv2.putText(blank, str(values[i]), (left_margin + 200, margin + int(margin * i) + int(thickness / 2)), font,
                    fontScale, fontColor, lineType)
    return blank


def test():
    values = [0.9, 0.6, 0.21]
    names = ['Fist', 'OK', 'Palm']

    while True:
        for i in range(len(values)):
            values[i] = randint(0, 100) / 100
        img = draw_classes_probabilities(values, 480, int(640/2), names)
        cv2.imshow("Classes probalities", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test()

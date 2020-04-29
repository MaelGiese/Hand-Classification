import os

# Use the CPU not the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from hand_tracking.utils import detector_utils as detector_utils
import cv2
import datetime
import argparse
import os
import keyboard

train_images = True

classes = ['Fist', 'OK', 'Palm']

dataset_path = 'hand_classification/Dataset'

# Si le répertoire n'existe pas le créer
if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)
    print("Directory ", dataset_path, " Created ")

if train_images:
    dataset_path = 'hand_classification/Dataset/Train/'
else:
    dataset_path = 'hand_classification/Dataset/Test/'

# Si le répertoire n'existe pas le créer
if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)
    print("Directory ", dataset_path, " Created ")

for hand_class in classes:
    # Si le répertoire n'existe pas le créer
    hand_class_path = dataset_path + hand_class + '/'
    if not os.path.exists(hand_class_path):
        os.mkdir(hand_class_path)
        print("Directory ", hand_class_path, " Created ")

dirs = os.listdir(dataset_path)

class_choice = -1
while class_choice not in range(0, len(dirs)):
    i = 0
    print('Choisir une pose de main :')
    for dir in dirs:
        print(str(i) + '. ' + dir)
        i = i + 1

    class_choice = int(input())

image_class = dirs[class_choice]

image_class_path = dataset_path + image_class + '/'

# Si le répertoire n'existe pas le créer
if not os.path.exists(image_class_path):
    os.mkdir(image_class_path)
    print("Directory ", image_class_path, " Created ")

dir_index = sum([len(dir) for r, dir, f in os.walk(image_class_path)])
set_path = image_class_path + 'SET_' + str(dir_index)

set_path = set_path + '/'

detection_graph, sess = detector_utils.load_inference_graph()
green = (77, 255, 9)
red = (255, 45, 56)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-sth',
        '--scorethreshold',
        dest='score_thresh',
        type=float,
        default=0.2,
        help='Score threshold for displaying bounding boxes')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=320,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=180,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=6,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    # max number of hands we want to detect/track
    num_hands_detect = 1

    cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)

    record = False
    color = green
    img_index = 0
    print('APPUYER SUR \'s\' POUR SAUVEGARDER UNE IMAGE !')
    while True:
        try:  # used try so that if user pressed other than the given key error will not be shown
            if keyboard.is_pressed('s'):
                record = True
                color = red
        except:
            break  # if user pressed a key other than the given key the loop will break

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ret, image_np = cap.read()
        image_np = cv2.flip(image_np, 1)
        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")

        # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
        # while scores contains the confidence for each of these boxes.
        # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

        boxes, scores = detector_utils.detect_objects(image_np,
                                                      detection_graph, sess)

        # draw bounding boxes on frame
        cropped_output = detector_utils.draw_box_on_image_and_return_cropped_image(num_hands_detect, args.score_thresh,
                                                                                   scores, boxes, im_width, im_height,
                                                                                   image_np, color)

        # Calculate Frames per second (FPS)
        num_frames += 1
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        fps = num_frames / elapsed_time

        if cropped_output is not None:
            cropped_output = cv2.cvtColor(cropped_output, cv2.COLOR_RGB2BGR)
            if args.display > 0:
                if record:
                    # Si le répertoire n'existe pas le créer
                    if not os.path.exists(set_path):
                        os.mkdir(set_path)
                        print("Directory ", set_path, " Created ")

                    cv2.imwrite(set_path + image_class + '_' + str(img_index) + '.png',
                                cropped_output)
                    print('Image' + str(img_index) + ' saved !')
                    img_index = img_index + 1

                cv2.namedWindow('Cropped', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Cropped', 450, 300)
                cv2.imshow('Cropped', cropped_output)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                if num_frames == 400:
                    num_frames = 0
                    start_time = datetime.datetime.now()

        if args.display > 0:
            # Display FPS on frame
            if args.fps > 0:
                detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                                 image_np)

            cv2.imshow('Single-Threaded Detection',
                       cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            print("frames processed: ", num_frames, "elapsed time: ",
                  elapsed_time, "fps: ", str(int(fps)))

        if record:
            record = False
            color = (77, 255, 9)
            file_index = dir_index + 1

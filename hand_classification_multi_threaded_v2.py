import os

# Use the CPU not the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy

import hand_classification_gui
from hand_tracking.utils import detector_utils as detector_utils
from hand_classification.utils import hand_classification_utils as classifier
import cv2
import tensorflow as tf
from multiprocessing import Queue, Pool
from hand_tracking.utils.detector_utils import WebcamVideoStream
import datetime
import argparse

frame_processed = 0
score_thresh = 0.2

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

classification_graph_path = 'hand_classification/models/hand_classification_model.h5'
classes = ['Fist', 'OK', 'Palm']

green = (77, 255, 9)
red = (255, 0, 0)
blue = (0, 0, 255)


# Create a worker thread that loads graph and
# does detection on images in an input queue and puts it on an output queue


def worker(input_q, output_q, cropped_output_q, classification_q, cap_params, frame_processed):
    print(">> loading frozen model for worker")
    detection_graph, sess = detector_utils.load_inference_graph()
    sess = tf.compat.v1.Session(graph=detection_graph)

    model, classification_graph, session = classifier.load_KerasGraph(classification_graph_path)

    color = green

    while True:
        # print("> ===== in worker loop, frame ", frame_processed)
        frame = input_q.get()
        if frame is not None:
            # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
            # while scores contains the confidence for each of these boxes.
            # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

            boxes, scores = detector_utils.detect_objects(
                frame, detection_graph, sess)
            # draw bounding boxes
            cropped_output = detector_utils.draw_box_on_image_and_return_cropped_image(
                cap_params['num_hands_detect'], cap_params["score_thresh"],
                scores, boxes, cap_params['im_width'], cap_params['im_height'],
                frame, color)

            if cropped_output is not None:
                class_prob = classifier.classify(model, classification_graph, session, cropped_output, IMAGE_WIDTH,
                                                 IMAGE_HEIGHT)
                classification_q.put(class_prob)

                max = (numpy.where(class_prob == numpy.amax(class_prob))[0])[0]

                if max == 0:
                    color = red
                else:
                    if max == 2:
                        color = green
                    else:
                        color = blue

            # add frame annotated with bounding box to queue
            cropped_output_q.put(cropped_output)
            output_q.put(frame)
            frame_processed += 1
        else:
            output_q.put(frame)
    sess.close()


def main(jump_q):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        type=int,
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-nhands',
        '--num_hands',
        dest='num_hands',
        type=int,
        default=1,
        help='Max number of hands to detect.')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=300,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=200,
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
        default=4,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=1,
        help='Size of the queue.')
    args = parser.parse_args()

    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    cropped_output_q = Queue(maxsize=args.queue_size)
    classification_q = Queue(1)

    video_capture = WebcamVideoStream(
        src=args.video_source, width=args.width, height=args.height).start()

    frame_processed = 0
    cap_params = {'im_width': video_capture.size()[0], 'im_height': video_capture.size()[1],
                  'score_thresh': score_thresh, 'num_hands_detect': args.num_hands}

    print(cap_params, args)

    # spin up workers to paralleize detection.
    pool = Pool(args.num_workers, worker,
                (input_q, output_q, cropped_output_q, classification_q, cap_params, frame_processed))

    start_time = datetime.datetime.now()
    num_frames = 0
    fps = 0
    index = 0

    #cv2.namedWindow('Multi-Threaded Detection', cv2.WINDOW_NORMAL)

    while True:
        frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        index += 1

        input_q.put(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        output_frame = output_q.get()
        cropped_output = cropped_output_q.get()

        class_proba = None
        try:
            class_proba = classification_q.get_nowait()
        except Exception:
            pass

        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        num_frames += 1
        fps = num_frames / elapsed_time

        if class_proba is not None:
            hand_classification_gui.draw_classes_probabilities(class_proba, classes)

            max = (numpy.where(class_proba == numpy.amax(class_proba))[0])[0]
            print(classes[max])

            if not jump_q.full():
                jump_q.put(max, False)
            else:
                jump_q.get_nowait()
                jump_q.put(max, False)

        if cropped_output is not None:
            cropped_output = cv2.cvtColor(cropped_output, cv2.COLOR_RGB2BGR)
            if args.display > 0:
                #cv2.namedWindow('Cropped', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Cropped', 450, 300)
                cv2.imshow('Cropped', cropped_output)
                # cv2.imwrite('image_' + str(num_frames) + '.png', cropped_output)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                if num_frames == 400:
                    num_frames = 0
                    start_time = datetime.datetime.now()
                else:
                    print("frames processed: ", index, "elapsed time: ",
                          elapsed_time, "fps: ", str(int(fps)))

        if output_frame is not None:
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
            if args.display > 0:
                if args.fps > 0:
                    detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                                     output_frame)
                cv2.imshow('Multi-Threaded Detection', output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                if num_frames == 400:
                    num_frames = 0
                    start_time = datetime.datetime.now()
                else:
                    print("frames processed: ", index, "elapsed time: ",
                          elapsed_time, "fps: ", str(int(fps)))
        else:
            # print("video end")
            break
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    fps = num_frames / elapsed_time
    print("fps", fps)
    pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(False)

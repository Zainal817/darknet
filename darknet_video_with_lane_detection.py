from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue
from OpenCV_Utils import *

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default="./data/challenge.mp4",
                        help="video source. If empty, uses ./data/challenge.mp4")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="./bin/yolov4-tiny.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./cfg/yolov4-tiny.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    return parser.parse_args()


def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))


def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def video_capture(frame_queue, darknet_image_queue):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        frame_queue.put(frame_rgb)
        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
        darknet_image_queue.put(darknet_image)
    return


def inference(darknet_image_queue, detections_queue, fps_queue):
    while cap.isOpened():
        try:
            darknet_image = darknet_image_queue.get(timeout = 1)
            prev_time = time.time()
            detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)
            detections_queue.put(detections)
            fps = int(1/(time.time() - prev_time))
            fps_queue.put(fps)
            print("FPS: {}".format(fps))
            darknet.print_detections(detections, args.ext_output)
        except:
            return

def imageProcessing(image):
    result = imageCopy(image)
    HLS = convertColor(result, cv2.COLOR_BGR2HLS)
    Y_lower = np.array([15, 52, 75])
    Y_upper = np.array([30, 190, 255])
    Y_BIN = rangeColor(HLS, Y_lower, Y_upper)
    W_lower = np.array([0, 200, 0])
    W_upper = np.array([180, 255, 255])
    W_BIN = rangeColor(HLS, W_lower, W_upper)
    result = addImage(Y_BIN, W_BIN)
    MORPH_ELLIPSE = imageMorphologyKernel(cv2.MORPH_ELLIPSE, 7)
    result = imageMorphologyEx(result, cv2.MORPH_CLOSE , MORPH_ELLIPSE)    
    MORPH_CROSS = imageMorphologyKernel(cv2.MORPH_CROSS, 3)
    result = imageMorphologyEx(result, cv2.MORPH_OPEN , MORPH_CROSS)
    result_line = imageMorphologyEx(result, cv2.MORPH_GRADIENT , MORPH_CROSS)
    height, width = image.shape[:2]
    src_pt1 = [int(width*0.4), int(height*0.65)]
    src_pt2 = [int(width*0.6), int(height*0.65)]
    src_pt3 = [int(width*0.9), int(height*0.9)]
    src_pt4 = [int(width*0.1), int(height*0.9)]
    roi_poly_02 = np.array([[tuple(src_pt1), tuple(src_pt2), tuple(src_pt3), tuple(src_pt4)]], dtype=np.int32)
    line_roi = polyROI(result_line, roi_poly_02)
    lines = houghLinesP(line_roi, 1, np.pi/180, 10, 5, 10)
    result, min_x_left, min_x_right = lineFitting(image, lines, (0, 0, 255), 5, 5. * np.pi / 180.)
    half_point = width * 0.5
    if abs((min_x_right - half_point) - (half_point -min_x_left )) > width * 0.1:
        print("change direction")
    return result

def drawing(frame_queue, detections_queue, fps_queue):
    random.seed(3)  # deterministic bbox colors
    video = set_saved_video(cap, args.out_filename, (width, height))
    while cap.isOpened():
        try:
            frame_resized = frame_queue.get(timeout = 1)
            detections = detections_queue.get(timeout = 1)
            fps = fps_queue.get()
            if frame_resized is not None:
                # detect and draw lines
                bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
                res = imageProcessing(bgr)
                frame_resized = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

                image = darknet.draw_boxes(detections, frame_resized, class_colors, width, height)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if args.out_filename is not None:
                    video.write(image)
                if not args.dont_show:
                    cv2.namedWindow('Inference', cv2.WINDOW_NORMAL)
                    cv2.imshow('Inference', image)
                if cv2.waitKey(1) == 27:
                    break
        except:
            cap.release()
            video.release()
            cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)

    args = parser()
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
        )
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)
    input_path = str2int(args.input)
    cap = cv2.VideoCapture(input_path)
    Thread(target=video_capture, args=(frame_queue, darknet_image_queue)).start()
    Thread(target=inference, args=(darknet_image_queue, detections_queue, fps_queue)).start()
    Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue)).start()

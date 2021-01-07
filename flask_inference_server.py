from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue
from flask import Flask, request, Response, make_response
import numpy as np

app = Flask(__name__) # API 서버 초기화

@app.route('/api/darknet/in_image_get_image/<thresh>', methods=['POST'])
def api_inference_image_get_image(thresh):
    r = request
    print(r)
    nparr = np.frombuffer(r.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        print("broken")
    thresh = float(thresh)
    result = function_inference_image_get_image(img, thresh)
    _, imgencoded = cv2.imencode('.jpg', result)
    return make_response(imgencoded.tobytes())

def function_inference_image_get_image(img, thresh):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    image = draw_boxes(detections, rgb, class_colors, width, height)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image
def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax
def draw_boxes(detections, image, colors, width, height):
    origin_height, origin_width = image.shape[0], image.shape[1]
    ratio_width = origin_width / width
    ratio_height = origin_height / height
    for label, confidence, bbox in detections:
        left, top, right, bottom = bbox2points(bbox)
        left = int(left * ratio_width)
        right = int(right * ratio_width)
        top = int(top * ratio_height)
        bottom = int(bottom * ratio_height)
        cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[label], 2)
    return image

def main(
        cfg = "./cfg/yolov4-tiny.cfg",
        data = "./cfg/coco.data",
        weights = "./bin/yolov4-tiny.weights"
        ):
    global network, class_names, class_colors, width, height, darknet_image

    network, class_names, class_colors = darknet.load_network(
        cfg,
        data,
        weights,
        batch_size=1
    )
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)
    width = int(width)
    height = int(height)
    print(width, height)
    ip = "192.168.55.1"
    app.run(host=ip, port=5000, debug=False)

if __name__ == '__main__':
    random.seed(3)  # deterministic bbox colors
    main()

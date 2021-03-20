# -*- coding: utf-8 -*-
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import time

import yolo_v3
import yolo_v3_tiny
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from utils import load_coco_names, draw_boxes, get_boxes_and_inputs, get_boxes_and_inputs_pb, non_max_suppression, \
                  load_graph, letter_box_image

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'input_img', '2.mp4', 'Input image')
tf.app.flags.DEFINE_string(
    'output_img', 'output.jpg', 'Output image')
tf.app.flags.DEFINE_string(
    'class_names', 'coco.names', 'File with class names')
tf.app.flags.DEFINE_string(
    'weights_file', 'yolov3.weights', 'Binary file with detector weights')
tf.app.flags.DEFINE_string(
    'data_format', 'NCHW', 'Data format: NCHW (gpu only) / NHWC')
tf.app.flags.DEFINE_string(
    'ckpt_file', './saved_model/model.ckpt', 'Checkpoint file')
tf.app.flags.DEFINE_string(
    'frozen_model', 'frozen_darknet_yolov3_model.pb', 'Frozen tensorflow protobuf model')
tf.app.flags.DEFINE_bool(
    'tiny', False, 'Use tiny version of YOLOv3')
tf.app.flags.DEFINE_bool(
    'spp', False, 'Use SPP version of YOLOv3')

tf.app.flags.DEFINE_integer(
    'size', 416, 'Image size')

tf.app.flags.DEFINE_float(
    'conf_threshold', 0.5, 'Confidence threshold')
tf.app.flags.DEFINE_float(
    'iou_threshold', 0.4, 'IoU threshold')

tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 1.0, 'Gpu memory fraction to use')

def main(argv=None):

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)

    config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
    )

    classes = load_coco_names(FLAGS.class_names)



    t0 = time.time()
    frozenGraph = load_graph(FLAGS.frozen_model)
    print("Loaded graph in {:.2f}s".format(time.time()-t0))

    boxes, inputs = get_boxes_and_inputs_pb(frozenGraph)

    with tf.Session(graph=frozenGraph, config=config) as sess:
        t0 = time.time()
        print(FLAGS.input_img)
        cap = cv2.VideoCapture(FLAGS.input_img)
        # cap = cv2.VideoCapture(0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        videoWriter = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (int(width), int(height)))
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                frame = cv2.flip(frame, 0)
                img = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                img_resized = letter_box_image(img, FLAGS.size, FLAGS.size, 128)
                img_resized = img_resized.astype(np.float32)
                detected_boxes = sess.run(
                    boxes, feed_dict={inputs: [img_resized]})
                filtered_boxes = non_max_suppression(detected_boxes,
                                                     confidence_threshold=FLAGS.conf_threshold,
                                                     iou_threshold=FLAGS.iou_threshold)
                print("Predictions found in {:.2f}s".format(time.time() - t0))

                draw_boxes(filtered_boxes, img, classes, (FLAGS.size, FLAGS.size), True)

                fimg = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
                cv2.imshow("show",fimg)
                videoWriter.write(fimg)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        videoWriter.release()

if __name__ == '__main__':
    tf.app.run()

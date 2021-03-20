# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import time
import cv2
import yolo_v3
import yolo_v3_tiny

from utils import load_coco_names, draw_boxes, get_boxes_and_inputs, get_boxes_and_inputs_pb, non_max_suppression, \
                  load_graph, letter_box_image

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'input_img', '', 'Input image')
tf.app.flags.DEFINE_string(
    'output_img', '', 'Output image')
tf.app.flags.DEFINE_string(
    'class_names', 'voc.names', 'File with class names')
tf.app.flags.DEFINE_string(
    'weights_file', 'yolov3.weights', 'Binary file with detector weights')
tf.app.flags.DEFINE_string(
    'data_format', 'NCHW', 'Data format: NCHW (gpu only) / NHWC')
tf.app.flags.DEFINE_string(
    'ckpt_file', './saved_model/model.ckpt', 'Checkpoint file')
tf.app.flags.DEFINE_string(
    'frozen_model', '', 'Frozen tensorflow protobuf model')
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
    'gpu_memory_fraction', 0.6, 'Gpu memory fraction to use')




def main(argv=None):

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)

    config = tf.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
    )


    
    # import sys
    # result=[]
    # with open("VOC2007/ImageSets/Main/test.txt",'r') as f:
	#     for line in f:
	# 	    result.append(f)
    # print(result)
    # with open("VOC2007/ImageSets/Main/test.txt", 'r') as f:
    #     line =[]
    #     while True:
    #         line = f.readline()     # 逐行读取
    #         if not line:
    #             break
    #         print(line)             # 这里加了 ',' 是为了避免 print 自动换行
    results = []
    f = open("VOC2007/ImageSets/Main/test.txt","r") 
    lines = f.readlines()      #读取全部内容 ，并以列表方式返回
    for line in lines:
        results.append(line.strip('\n').split(',')[0])

    # if FLAGS.frozen_model:

    #     t0 = time.time()
    #     frozenGraph = load_graph(FLAGS.frozen_model)
    #     print("Loaded graph in {:.2f}s".format(time.time()-t0))

    #     boxes, inputs = get_boxes_and_inputs_pb(frozenGraph)

    #     with tf.Session(graph=frozenGraph, config=config) as sess:
    #         t0 = time.time()
    #         detected_boxes = sess.run(
    #             boxes, feed_dict={inputs: [img_resized]})

    # else:
        # if FLAGS.tiny:
        #     model = yolo_v3_tiny.yolo_v3_tiny
        # elif FLAGS.spp:
        #     model = yolo_v3.yolo_v3_spp
        # else:
    model = yolo_v3.yolo_v3
    classes = load_coco_names(FLAGS.class_names)
    boxes, inputs = get_boxes_and_inputs(model, len(classes), FLAGS.size, FLAGS.data_format)
    saver = tf.train.Saver(var_list=tf.global_variables(scope='detector'))
    with tf.Session(config=config) as sess:
        t0 = time.time()
        saver.restore(sess, FLAGS.ckpt_file)
        print('Model restored in {:.2f}s'.format(time.time()-t0))

        t0 = time.time()
        # file_list = os.listdir('input/')
        
        for file in results:
            try:
                print('VOC2007/JPEGImages/'+str(file)+'.jpg')
                image = cv2.imread('VOC2007/JPEGImages/'+str(file)+'.jpg')
                print(image.shape)
                
                img = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)) 
                # img = Image.open('VOC2007/JPEGImages/'+str(file)+'.jpg')
                img_resized = letter_box_image(img, FLAGS.size, FLAGS.size, 128)
                img_resized = img_resized.astype(np.float32)
                detected_boxes = sess.run(
                    boxes, feed_dict={inputs: [img_resized]})

                filtered_boxes = non_max_suppression(detected_boxes,
                                                    confidence_threshold=FLAGS.conf_threshold,
                                                    iou_threshold=FLAGS.iou_threshold)
                print("Predictions found in {:.2f}s".format(time.time() - t0))

                draw_boxes(filtered_boxes, img, classes, (FLAGS.size, FLAGS.size), True)

                img.save('output/'+file+'.jpg')
            except ValueError:
                pass


if __name__ == '__main__':
    tf.app.run()

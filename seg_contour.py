import os
from io import BytesIO

import numpy as np
from PIL import Image
import cv2

import tensorflow as tf
import sys
import datetime
import pdb


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        graph_def = tf.GraphDef.FromString(
            open(tarball_path + "/frozen_inference_graph.pb", "rb").read())

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

    Args:
      image: Numpy RGB image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
        start = datetime.datetime.now()

        height, width = image.shape[0:2]
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = cv2.resize(image, target_size, interpolation =  cv2.INTER_LINEAR_EXACT)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [resized_image.astype(np.uint8)]})
        seg_map = batch_seg_map[0]

        end = datetime.datetime.now()

        diff = end - start
        print("Time taken to evaluate segmentation is : " + str(diff))
        seg_height, seg_width = seg_map.shape
        seg_map = seg_map.astype(np.uint8)
        im2, contours, hierarchy = cv2.findContours(seg_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            if cv2.contourArea(contour) < (seg_height * seg_width * 0.2):
                #seg_map = cv2.drawContours(seg_map, contour, -1, (255, 255, 255), 1)
                seg_map = cv2.fillPoly(seg_map, pts = [contour], color=0)
        seg_map = np.expand_dims(seg_map.astype(np.uint8), -1)
        seg_map = np.concatenate([seg_map, seg_map, seg_map], -1)
        seg_map = cv2.resize(seg_map, (width, height), interpolation =  cv2.INTER_LINEAR_EXACT)
        return image, seg_map

def drawSegment_cv(baseImg, segMat, vid = False):
    clearIdxs = segMat == 0
    baseImg[clearIdxs] = 0
    if not vid:
        cv2.imwrite(outputFilePath, cv2.cvtColor(baseImg, cv2.COLOR_RGB2BGR))
    else:
        return cv2.cvtColor(baseImg, cv2.COLOR_RGB2BGR)


inputFilePath = sys.argv[1]
outputFilePath = sys.argv[2]

if inputFilePath is None or outputFilePath is None:
    print(
        "Bad parameters. Please specify input file path and output file path")
    exit()

modelType = "mobile_net_model"
if len(sys.argv) > 3 and sys.argv[3] == "1":
    modelType = "xception_model"
if len(sys.argv) > 3 and sys.argv[3] == "2":
    modelType = "deeplabv3_mnv2_pascal_trainval"
if len(sys.argv) > 3 and sys.argv[3] == "3":
    modelType = "optimized_mobilenet"
if len(sys.argv) > 3 and sys.argv[3] == "4":
    modelType = "xception_optimized"
if len(sys.argv) > 3 and sys.argv[3] == "5":
    modelType = "mobile_net_model"
MODEL = DeepLabModel(modelType)
print('model loaded successfully : ' + modelType)


def run_visualization(filepath):
    """Inferences DeepLab model and visualizes result."""

    try:
        print("Trying to open : " + sys.argv[1])
        if sys.argv[4] == '--video':
            cap = cv2.VideoCapture(filepath)
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            thresh_area = frame_height * frame_width * 0.9
            out = cv2.VideoWriter('result.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
            i = 1
            while cap.isOpened():
                ret, frame = cap.read()
                if ret is True and i%2 != 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    resized_im, seg_map = MODEL.run(frame_rgb)
                    to_write_frame = drawSegment_cv(resized_im, seg_map, vid = True)
                    out.write(to_write_frame)
                    cv2.imshow('frame',to_write_frame)
                elif i%2 == 0 and ret is True:
                    out.write(to_write_frame)             
                    cv2.imshow('frame',to_write_frame)           
                    # Press Q on keyboard to stop recording
                else:
                    break
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        elif sys.argv[5] == '--image':
            original_im = cv2.imread(filepath)
            original_im = cv2.cvtColor(original_im, cv2.COLOR_BGR2RGB)
            print('running deeplab on image %s...' % filepath)
            for i in range(10):
                resized_im, seg_map = MODEL.run(original_im)
            drawSegment_cv(resized_im, seg_map)
    except IOError:
        print('Cannot retrieve the input file. Please check file: ' + filepath)
        return
    # vis_segmentation(resized_im, seg_map)
    # drawSegment(resized_im, seg_map)
run_visualization(inputFilePath)
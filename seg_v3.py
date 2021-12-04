import os
import numpy as np
from PIL import Image
import cv2

import tensorflow as tf
import sys
import datetime
import collections
import argparse


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
        self.prev_seg_maps = collections.deque([], maxlen=24)
        self.prev_frames = collections.deque([], maxlen=24)
        self.frame_count = 0

    def compute_dnn_seg_map(self, resized_image):
        """Compute the seg map using the dnn model"""

        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [resized_image.astype(np.uint8)]})
        seg_map = batch_seg_map[0]
        return seg_map

    def run(self, image):
        """Runs inference on a single image.

        Args:
        image: Numpy RGB image.

        Returns:
        resized_image: RGB image resized from original input image.
        seg_map: Segmentation map of `resized_image`.
        """

        start = datetime.datetime.now()

        image_orig = image.copy()

        if len(self.prev_frames) > 0:
            blur = cv2.GaussianBlur(self.prev_frames[-1],(51,51),51)
            # import pdb; pdb.set_trace()
            image = np.where(np.repeat(np.expand_dims(self.prev_seg_maps[-1], -1), 3, -1) == 0, blur, image)
            cv2.imshow('blur', image)

        height, width = image.shape[0:2]
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = cv2.resize(image, target_size)



        if self.frame_count % 1 == 0:
            seg_map = self.compute_dnn_seg_map(resized_image)
        else:
            seg_map = self.prev_seg_maps[-1]

        end = datetime.datetime.now()

        diff = end - start
        print("Time taken to evaluate segmentation is : " + str(diff))
        seg_map = seg_map.astype(np.uint8)

        seg_map = cv2.resize(seg_map, (width, height))
        self.prev_seg_maps.append(seg_map)
        self.prev_frames.append(image_orig)
        self.frame_count += 1

        return image, seg_map


def drawSegment_cv(baseImg, segMat):
    """Segment out the seg map from the image."""

    segMat3 = np.repeat(np.expand_dims(segMat, -1), 3, -1)
    clearIdxs = segMat3 == 0
    baseImg[clearIdxs] = 255

    return baseImg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_fpath')
    parser.add_argument('--output_fpath', default='output.mp4')
    parser.add_argument('--model_type', default='xo', required=False)

    args = parser.parse_args()

    if args.input_fpath == '0':
        args.input_fpath = 0

    if args.model_type == 'xo':
       args.model_type = "xception_optimized"
    elif args.model_type == 'x':
        args.model_type = "xception_model"
    elif args.model_type == 'mo':
        args.model_type = "optimized_mobilenet"
    elif args.model_type == 'm':
        args.model_type = "mobile_net_model"

    if args.output_fpath == 'disabled':
        args.output_fpath = None

    return args


def run_inference(args, MODEL):

    print("Trying to open : " + sys.argv[1])
    cap = cv2.VideoCapture(args.input_fpath)
    if not cap:
        print('Invalid file path')
        return

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    if args.output_fpath:
        out = cv2.VideoWriter(args.output_fpath,
                              cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
        print('Writing output to', args.output_fpath)

    while cap.isOpened():
        ret, frame = cap.read()
        print(ret, frame.shape)
        if ret is None:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_im, seg_map = MODEL.run(frame_rgb)
        to_write_frame = drawSegment_cv(resized_im, seg_map)
        to_write_frame = cv2.cvtColor(to_write_frame, cv2.COLOR_RGB2BGR)

        cv2.imshow('frame', to_write_frame)
        cv2.imshow('frameseg', seg_map)

        if args.output_fpath:
            out.write(to_write_frame)

        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    """Inferences DeepLab model and visualizes result."""

    args = parse_args()

    MODEL = DeepLabModel(args.model_type)
    print('model loaded successfully :', args.model_type)

    run_inference(args, MODEL)


if __name__ == '__main__':
    main()

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

        # State
        self.nn_seg_map = None
        self.nn_seg_map_area = None
        self.nn_seg_map_centroid = None
        self.frame_count = 0

        self.prev_seg_maps = collections.deque([], maxlen=10)
        self.prev_frames = collections.deque([], maxlen=10)

        # Thresholds
        self.thresh_area_diff = 0.5
        self.thresh_centroid_diff = 0.3
        self.thresh_color_diff = 200
        self.follow_kernel_size = np.ones((3,3),np.uint8)

    def compute_seg_centroid(self, seg_map):
        """Compute centroid of single channel seg map"""

        try:
            # calculate moments of binary image
            M = cv2.moments((seg_map == 0).astype(np.uint8) * 255)

            # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

        except ZeroDivisionError:

            cY, cX = seg_map.shape
            cY, cX = cY//2, cX//2

        return (cX, cY)


    def compute_dnn_seg_map(self, resized_image):
        """Compute the seg map using the dnn model"""

        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [resized_image.astype(np.uint8)]})
        seg_map = batch_seg_map[0]

        # import pdb; pdb.set_trace()
        # seg_map = (seg_map > seg_map.mean()).astype(np.uint8)
        self.nn_seg_map = seg_map
        self.nn_seg_map_area = np.sum(seg_map > 0)
        self.nn_seg_map_centroid = self.compute_seg_centroid(seg_map)

        return seg_map

    def should_update_seg_map(self, seg_map):

        if self.frame_count == 0:
            print('First frame')
            return True

        new_area = np.sum(seg_map > 0)
        if abs(new_area - self.nn_seg_map_area) / max(new_area, self.nn_seg_map_area) > self.thresh_area_diff:
            print('Area has changed', new_area, self.nn_seg_map_area, new_area/self.nn_seg_map_area)
            # import pdb; pdb.set_trace()
            return True

        # TODO cache frame diag len values
        frame_h, frame_w = seg_map.shape
        frame_diag_len = (frame_h * frame_h) + (frame_w * frame_w)

        new_centroid_x, new_centroid_y = self.compute_seg_centroid(seg_map)
        old_centroid_x, old_centroid_y = self.nn_seg_map_centroid
        centroid_dx = new_centroid_x - old_centroid_x
        centroid_dy = new_centroid_y - old_centroid_y
        centroid_diff_len = (centroid_dx * centroid_dx) + (centroid_dy * centroid_dy)
        if centroid_diff_len / frame_diag_len > self.thresh_centroid_diff:
            print('Centroid has changed')
            # import pdb; pdb.set_trace()
            return True

        return False


    def seg_map_follow_person(self, current_frame):
        """Grow the segmentation map based on the color diffs."""

        if len(self.prev_seg_maps) == 0:  # First frame, so compute from dnn directly
            return None

        prev_seg_map = self.prev_seg_maps[-1]
        prev_person_map = prev_seg_map != 0

        # Compute color diffs in hsv
        current_frame_hue = cv2.cvtColor(current_frame, cv2.COLOR_RGB2HSV)[:, :, 0]
        prev_frame = self.prev_frames[-1]
        prev_frame_hue = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2HSV)[:, :, 0]  # TODO save only hues

        current_person_values = current_frame_hue * prev_person_map
        current_seg_values = current_frame_hue * (prev_seg_map == 0)

        prev_person_values = prev_frame_hue * prev_person_map
        prev_seg_values = prev_frame_hue * (prev_seg_map == 0)

        # Find differences
        diff_seg = np.abs(current_seg_values - prev_seg_values) > self.thresh_color_diff
        diff_person = np.abs(current_person_values - prev_person_values) > self.thresh_color_diff

        # Only take diffferences at the segmentation border
        diff_seg = cv2.morphologyEx(prev_person_map.astype(np.uint8), cv2.MORPH_GRADIENT, self.follow_kernel_size) * diff_seg.astype(np.uint8)
        diff_person = cv2.morphologyEx(prev_person_map.astype(np.uint8), cv2.MORPH_GRADIENT, self.follow_kernel_size) * diff_person.astype(np.uint8)

        # import pdb; pdb.set_trace()
        updated_seg_map = prev_seg_map.copy()
        updated_seg_map[diff_person.astype(np.bool)] = 0  # Person has exited
        updated_seg_map[diff_seg.astype(np.bool)] = 15  # Person has entered

        # updated_seg_map[diff_person.astype(np.bool)] -= 1  # Person has exited
        # updated_seg_map[diff_seg.astype(np.bool)] += 1  # Person has entered


        return updated_seg_map


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
        resized_image = cv2.resize(image, target_size)

        # TODO grow seg map to follow person
        seg_map = self.seg_map_follow_person(resized_image)

        if self.should_update_seg_map(seg_map):
            seg_map = self.compute_dnn_seg_map(resized_image)

        end = datetime.datetime.now()

        diff = end - start
        print("Time taken to evaluate segmentation is : " + str(diff))
        seg_map = seg_map.astype(np.uint8)

        self.prev_seg_maps.append(seg_map)
        self.prev_frames.append(resized_image)
        self.frame_count += 1

        return image, seg_map


def drawSegment_cv(baseImg, segMat):
    """Segment out the seg map from the image."""

    segMat = cv2.resize(segMat, (baseImg.shape[1], baseImg.shape[0]))
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
        if not ret:
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

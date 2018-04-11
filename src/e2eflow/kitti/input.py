import os
import sys

import numpy as np
import tensorflow as tf
import random

from ..core.input import read_png_image, read_images_from_disk, Input
from ..core.augment import random_crop
import scipy.misc
from .pose_evaluation_utils import *


def _read_flow(filenames, num_epochs=None):
    """Given a list of filenames, constructs a reader op for ground truth."""
    filename_queue = tf.train.string_input_producer(filenames,
        shuffle=False, capacity=len(filenames), num_epochs=num_epochs)
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    gt_uint16 = tf.image.decode_png(value, dtype=tf.uint16)
    gt = tf.cast(gt_uint16, tf.float32)
    flow = (gt[:, :, 0:2] - 2 ** 15) / 64.0
    mask = gt[:, :, 2:3]
    return flow, mask

def _read_raw_intrinsics_gt(filenames):
    # filenames : list('../data/kitti_raw/dates/dates_drives/image_cam/data/**.png', ..)
    intrinsics = []
    filenames = [f for f in filenames if f.endswith('.txt')]
    for file in filenames:
        filedata = {}
        with open(file, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    filedata[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        try :
            P_rect = np.reshape(filedata['P_rect_00'], (3, 4))
        except :
            P_rect = np.reshape(filedata['P0'], (3, 4))
        P_rect_tf = tf.constant(P_rect[:3,:3])
        intrinsics.append(P_rect_tf)
    return intrinsics

class KITTIInput(Input):
    def __init__(self, data, batch_size, dims, *,
                 num_threads=1, normalize=True,
                 skipped_frames=False):
        super().__init__(data, batch_size, dims, num_threads=num_threads,
                         normalize=normalize, skipped_frames=skipped_frames)

    def _preprocess_flow(self, gt):
        flow, mask = gt
        height, width = self.dims
        # Reshape to tell tensorflow we know the size statically
        flow = tf.reshape(self._resize_crop_or_pad(flow), [height, width, 2])
        mask = tf.reshape(self._resize_crop_or_pad(mask), [height, width, 1])
        return flow, mask

    def _input_flow(self, flow_dir, hold_out_inv):
        flow_dir_occ = os.path.join(self.data.current_dir, flow_dir, 'flow_occ')
        flow_dir_noc = os.path.join(self.data.current_dir, flow_dir, 'flow_noc')
        filenames_gt_occ = []
        filenames_gt_noc = []
        flow_files_occ = os.listdir(flow_dir_occ)
        flow_files_occ.sort()
        flow_files_noc = os.listdir(flow_dir_noc)
        flow_files_noc.sort()

        if hold_out_inv is not None:
            random.seed(0)
            random.shuffle(flow_files_noc)
            flow_files_noc = flow_files_noc[:hold_out_inv]

            random.seed(0)
            random.shuffle(flow_files_occ)
            flow_files_occ = flow_files_occ[:hold_out_inv]

        assert len(flow_files_noc) == len(flow_files_occ)

        for i in range(len(flow_files_occ)):
            filenames_gt_occ.append(os.path.join(flow_dir_occ,
                                                 flow_files_occ[i]))
            filenames_gt_noc.append(os.path.join(flow_dir_noc,
                                                 flow_files_noc[i]))

        flow_occ, mask_occ = self._preprocess_flow(
            _read_flow(filenames_gt_occ, 1))
        flow_noc, mask_noc = self._preprocess_flow(
            _read_flow(filenames_gt_noc, 1))
        return flow_occ, mask_occ, flow_noc, mask_noc

    def _input_train(self, image_dir, flow_dir, hold_out_inv=None):
        input_shape, im1, im2 = self._input_images(image_dir, hold_out_inv)
        flow_occ, mask_occ, flow_noc, mask_noc = self._input_flow(flow_dir, hold_out_inv)
        return tf.train.batch(
            [im1, im2, input_shape, flow_occ, mask_occ, flow_noc, mask_noc],
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            allow_smaller_final_batch=True)

    def _input_pose(self, pose_dir, hold_out_inv):
        test_sequences = ['09_full.txt']

        poses = []
        for sequence in test_sequences:
            pose_dir = os.path.join(self.data.current_dir, pose_dir, sequence)
            with open(pose_dir) as f:
                for line in f:
                    line = line.replace(",", " ").replace("\t"," ").rstrip("\n")
                    pose = np.fromstring(line, dtype=float, sep=' ')
                    pose = tf.cast(pose, dtype=tf.float32)
                    poses.append(pose)
        return poses[1:]

    def _input_odometry(self, image_dir, pose_dir, shift, hold_out_inv=None):
        test_sequences = ['09']
        filenames_1 = []
        filenames_2 = []
        for sequence in test_sequences:
            image_02_folder = os.path.join(self.data.current_dir, image_dir, sequence, 'image_2/')
            image_files = os.listdir(image_02_folder)

            image_files.sort()

            for i in range(len(image_files) - 1):
                filenames_1.append(os.path.join(image_02_folder, image_files[i]))
                filenames_2.append(os.path.join(image_02_folder, image_files[i + 1]))

        poses = self._input_pose(pose_dir, hold_out_inv)

        input_queue = tf.train.slice_input_producer([filenames_1, filenames_2, poses], capacity=1590, shuffle=False)
        input_1, input_2 = read_images_from_disk(input_queue[0:2])
        pose = input_queue[2]

        image_1 = self._preprocess_image(input_1)
        image_2 = self._preprocess_image(input_2)
        input_shape = tf.shape(input_1)

        return tf.train.batch(
            [image_1, image_2, input_shape, pose],
            batch_size=self.batch_size,
            num_threads=self.num_threads,
            allow_smaller_final_batch=True)

    def input_train_gt(self, hold_out):
        img_dirs = ['data_scene_flow/training/image_2',
                    'data_stereo_flow/training/colored_0']
        gt_dirs = ['data_scene_flow/training/flow_occ',
                   'data_stereo_flow/training/flow_occ']
        calib_dirs = ['data_scene_flow/training/calib',
                   'data_stereo_flow/training/calib']

        height, width = self.dims

        filenames = []
        for img_dir, gt_dir, calib_dir in zip(img_dirs, gt_dirs, calib_dirs):
            dataset_filenames = []
            img_dir = os.path.join(self.data.current_dir, img_dir)
            gt_dir = os.path.join(self.data.current_dir, gt_dir)
            calib_dir = os.path.join(self.data.current_dir, calib_dir)
            img_files = os.listdir(img_dir)
            gt_files = os.listdir(gt_dir)
            calib_files = os.listdir(calib_dir)
            img_files.sort()
            gt_files.sort()
            calib_files.sort()
            assert len(img_files) % 2 == 0 and len(img_files) / 2 == len(gt_files)

            for i in range(len(gt_files)):
                fn_im1 = os.path.join(img_dir, img_files[2 * i])
                fn_im2 = os.path.join(img_dir, img_files[2 * i + 1])
                fn_gt = os.path.join(gt_dir, gt_files[i])
                fn_calib = os.path.join(calib_dir, calib_files[i])
                dataset_filenames.append((fn_im1, fn_im2, fn_gt, fn_calib))

            random.seed(0)
            random.shuffle(dataset_filenames)
            dataset_filenames = dataset_filenames[hold_out:]
            filenames.extend(dataset_filenames)

        random.seed(0)
        random.shuffle(filenames)

        #shift = shift % len(filenames)
        #filenames_ = list(np.roll(filenames, shift))

        fns_im1, fns_im2, fns_gt, fns_calib = zip(*filenames)
        fns_im1 = list(fns_im1)
        fns_im2 = list(fns_im2)
        fns_gt = list(fns_gt)
        fns_calib = list(fns_calib)

        intrinsics = _read_raw_intrinsics_gt(fns_calib)
        input_queue = tf.train.slice_input_producer([fns_im1, fns_im2, intrinsics],
                                                    shuffle=False)
        im1, im2 = read_images_from_disk(input_queue[0:2])
        intrinsic = input_queue[2]
        flow_gt, mask_gt = _read_flow(fns_gt)

        gt_queue = tf.train.string_input_producer(fns_gt,
            shuffle=False, capacity=len(fns_gt), num_epochs=None)
        reader = tf.WholeFileReader()
        _, gt_value = reader.read(gt_queue)
        gt_uint16 = tf.image.decode_png(gt_value, dtype=tf.uint16)
        gt = tf.cast(gt_uint16, tf.float32)

        im1, im2, gt, intrinsic = random_crop([im1, im2, gt],
                                   [height, width, 3], intrinsic=intrinsic)
        flow_gt = (gt[:, :, 0:2] - 2 ** 15) / 64.0
        mask_gt = gt[:, :, 2:3]

        if self.normalize:
            im1 = self._normalize_image(im1)
            im2 = self._normalize_image(im2)

        return tf.train.batch(
            [im1, im2, flow_gt, mask_gt, intrinsic],
            batch_size=self.batch_size,
            num_threads=self.num_threads)

    def input_train_2015(self, hold_out_inv=None):
        return self._input_train('data_scene_flow/training/image_2',
                                 'data_scene_flow/training',
                                 hold_out_inv)

    def input_test_2015(self, hold_out_inv=None):
        return self._input_test('data_scene_flow/testing/image_2', hold_out_inv)

    def input_train_2012(self, hold_out_inv=None):
        return self._input_train('data_stereo_flow/training/colored_0',
                                 'data_stereo_flow/training',
                                 hold_out_inv)

    def input_test_2012(self, hold_out_inv=None):
        return self._input_test('data_stereo_flow/testing/colored_0', hold_out_inv)

    def input_odometry(self, hold_out_inv=None):
        return self._input_odometry('kitti_odom/sequences',
                                 'kitti_odom/gt_pose/ground_truth',
                                 hold_out_inv)
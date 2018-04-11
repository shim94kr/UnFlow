import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from .augment import random_affine, random_photometric
from .flow_util import flow_to_color
from .util import resize_area, resize_bilinear
from .losses import compute_losses, create_border_mask
from ..ops import downsample
from .image_warp import image_warp
from .flownet import flownet, FLOW_SCALE
from .augment import get_multi_scale_intrinsics


POSE_SCALE = 0.0001
# REGISTER ALL POSSIBLE LOSS TERMS
LOSSES = ['occ', 'sym', 'sym_pose', 'fb', 'grad', 'ternary', 'photo', 'smooth_1st', 'smooth_pose_2nd', 'smooth_2nd', 'epipolar', 'pose_scale']


def _track_loss(op, name):
    tf.add_to_collection('losses', tf.identity(op, name=name))


def _track_image(op, name):
    name = 'train/' + name
    tf.add_to_collection('train_images', tf.identity(op, name=name))


def unsupervised_loss(batch, params, normalization=None, augment=True,
                      return_flow=False, return_pose=False):
    intrinsics1 = []
    if len(batch) == 3:
        im1, im2, intrinsics1 = batch
    else:
        im1, im2 = batch
    im1 = im1 / 255.0
    im2 = im2 / 255.0
    im_shape = tf.shape(im1)[1:3]
    channel_mean = tf.constant(normalization[0]) / 255.0
    # -------------------------------------------------------------------------
    # Data & mask augmentation
    border_mask = create_border_mask(im1, 0.1)
    if augment:
        if len(batch) == 3:
            im1_geo, im2_geo, border_mask_global, intrinsics1 = random_affine(
                [im1, im2, border_mask],
                horizontal_flipping=False,
                min_scale=0.9, max_scale=1.1,
                intrinsics=intrinsics1
                )
            intrinsics2 = intrinsics1
            im2_geo, border_mask_local, intrinsics2 = random_affine(
                [im2_geo, border_mask],
                horizontal_flipping=False,
                min_scale=0.9, max_scale=1.1,
                intrinsics=intrinsics2
                )
            border_mask = border_mask_local * border_mask_global
            intrinsics1 = get_multi_scale_intrinsics(intrinsics1, 7)
            intrinsics2 = get_multi_scale_intrinsics(intrinsics2, 7)
        else:
            im1_geo, im2_geo, border_mask_global = random_affine(
                [im1, im2, border_mask],
                horizontal_flipping=True,
                min_scale=0.9, max_scale=1.1
                )
            # augment locally
            im2_geo, border_mask_local = random_affine(
                [im2_geo, border_mask],
                min_scale=0.9, max_scale=1.1
                )
            border_mask = border_mask_local * border_mask_global

        im1_photo, im2_photo = random_photometric(
            [im1_geo, im2_geo],
            noise_stddev=0.04, min_contrast=-0.3, max_contrast=0.3,
            brightness_stddev=0.02, min_colour=0.9, max_colour=1.1,
            min_gamma=0.7, max_gamma=1.5)

        _track_image(im1_photo, 'augmented1')
        _track_image(im2_photo, 'augmented2')
    else:
        im1_geo, im2_geo = im1, im2
        im1_photo, im2_photo = im1, im2

    # Images for loss comparisons with values in [0, 1] (scale to original using * 255)
    im1_norm = im1_geo
    im2_norm = im2_geo
    # Images for neural network input with mean-zero values in [-1, 1]
    im1_photo = im1_photo - channel_mean
    im2_photo = im2_photo - channel_mean

    flownet_spec = params.get('flownet', 'S')
    full_resolution = params.get('full_res')
    pose_prediction = params.get('pose_pred')
    train_all = params.get('train_all')

    flows_fw, flows_bw, flows2_fw, flows2_bw, poses_fw, poses_bw = flownet(im1_photo, im2_photo,
                                                flownet_spec=flownet_spec,
                                                full_resolution=full_resolution,
                                                pose_prediction=pose_prediction,
                                                backward_flow=True,
                                                train_all=train_all)

    flows_fw = flows_fw[-1]
    flows_bw = flows_bw[-1]
    if pose_prediction:
        flows2_fw = flows2_fw[-1]
        flows2_bw = flows2_bw[-1]
        poses_fw = poses_fw[-1]
        poses_bw = poses_bw[-1]

    # -------------------------------------------------------------------------
    # Losses
    layer_weights = [12.7, 4.35, 3.9, 3.4, 1.1]
    layer_patch_distances = [3, 2, 2, 1, 1]
    if full_resolution:
        layer_weights = [12.7, 5.5, 5.0, 4.35, 3.9, 3.4, 1.1]
        layer_patch_distances = [3, 3] + layer_patch_distances
        im1_s = im1_norm
        im2_s = im2_norm
        mask_s = border_mask
        final_flow_scale = FLOW_SCALE * 4
        final_flow_fw = flows_fw[0] * final_flow_scale
        final_flow_bw = flows_bw[0] * final_flow_scale
    else:
        im1_s = downsample(im1_norm, 4)
        im2_s = downsample(im2_norm, 4)
        mask_s = downsample(border_mask, 4)
        final_flow_scale = FLOW_SCALE
        final_flow_fw = tf.image.resize_bilinear(flows_fw[0], im_shape) * final_flow_scale * 4
        final_flow_bw = tf.image.resize_bilinear(flows_bw[0], im_shape) * final_flow_scale * 4

    combined_losses = dict()
    combined_loss = 0.0
    for loss in LOSSES:
        combined_losses[loss] = 0.0

    if params.get('pyramid_loss'):
        flow_enum = enumerate(zip(flows_fw, flows_bw))
    else:
        flow_enum = [(0, (flows_fw[0], flows_bw[0]))]

    mean_pose_fw = None
    mean_pose_bw = None
    for i, flow_pair in flow_enum:
        layer_name = "loss" + str(i + 2)

        flow_scale = final_flow_scale / (2 ** i)

        with tf.variable_scope(layer_name):
            layer_weight = layer_weights[i]
            flow_fw_s, flow_bw_s = flow_pair
            mask_occlusion = params.get('mask_occlusion', '')
            assert mask_occlusion in ['fb', 'disocc', '', 'both']
            """
            if "P" in flownet_spec:
                poses_fw[i] = POSE_SCALE * poses_fw[i]
                poses_bw[i] = POSE_SCALE * poses_bw[i]
            """
            if len(batch) == 3:
                losses, pose_fw_all, pose_bw_all = compute_losses(im1_s, im2_s,
                            flow_fw_s * flow_scale, flow_bw_s * flow_scale,
                            poses_fw[i], poses_bw[i], intrinsics1[:, i + 2, :, :], intrinsics2[:, i + 2, :, :],
                            flows2_fw[i], flows2_bw[i],
                            border_mask=mask_s if params.get('border_mask') else None,
                            mask_occlusion=mask_occlusion,
                            data_max_distance=layer_patch_distances[i])
                if mean_pose_fw is None :
                    mean_pose_fw = tf.reduce_mean(pose_fw_all, axis=[1, 2])
                    mean_pose_bw = tf.reduce_mean(pose_bw_all, axis=[1, 2])
                else:
                    mean_pose_fw = tf.concat([mean_pose_fw, tf.reduce_mean(pose_fw_all, axis=[1, 2])], axis=0)
                    mean_pose_bw = tf.concat([mean_pose_bw, tf.reduce_mean(pose_bw_all, axis=[1, 2])], axis=0)
            else :
                losses = compute_losses(im1_s, im2_s,
                                    flow_fw_s * flow_scale, flow_bw_s * flow_scale,
                                    border_mask=mask_s if params.get('border_mask') else None,
                                    mask_occlusion=mask_occlusion,
                                    data_max_distance=layer_patch_distances[i])


            layer_loss = 0.0

            for loss in LOSSES:
                weight_name = loss + '_weight'
                if params.get(weight_name):
                    _track_loss(losses[loss], loss)
                    layer_loss += params[weight_name] * losses[loss]
                    combined_losses[loss] += layer_weight * losses[loss]

            combined_loss += layer_weight * layer_loss

            im1_s = downsample(im1_s, 2)
            im2_s = downsample(im2_s, 2)
            mask_s = downsample(mask_s, 2)

    if len(batch) == 3:
        tf.add_to_collection('poses', tf.identity(mean_pose_fw, name='pose_fw'))
        tf.add_to_collection('poses', tf.identity(mean_pose_bw, name='pose_bw'))
    regularization_loss = tf.losses.get_regularization_loss()
    final_loss = combined_loss + regularization_loss

    _track_loss(final_loss, 'loss/combined')

    for loss in LOSSES:
        _track_loss(combined_losses[loss], 'loss/' + loss)
        weight_name = loss + '_weight'
        if params.get(weight_name):
            weight = tf.identity(params[weight_name], name='weight/' + loss)
            tf.add_to_collection('params', weight)

    if return_flow:
        return final_loss, final_flow_fw, final_flow_bw
    elif return_pose:
        return final_loss, poses_fw, poses_bw, final_flow_fw, final_flow_bw
    else:
        return final_loss

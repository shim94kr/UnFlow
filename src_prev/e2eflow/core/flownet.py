import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers

from ..ops import correlation
from .image_warp import image_warp
import numpy as np

from .flow_util import flow_to_color
from ..ops import backward_warp, forward_warp
from .losses import occlusion, DISOCC_THRESH, create_outgoing_mask

FLOW_SCALE = 5.0
POSE_SCALING = 0.001


def flownet(im1, im2, flownet_spec='S', full_resolution=False, train_all=False,
            backward_flow=False, pose_prediction = False):
    num_batch, height, width, _ = tf.unstack(tf.shape(im1))
    flownet_num = len(flownet_spec)
    assert flownet_num > 0
    flows_fw = []
    flows_bw = []
    flows2_fw = []
    flows2_bw = []
    poses_fw = []
    poses_bw = []
    for i, name in enumerate(flownet_spec):
        assert name in ('C', 'c', 'S', 's', 'P', 'p')
        channel_mult = 1 if name in ('C', 'S', 'P') else 3 / 8
        full_res = full_resolution and i == flownet_num - 1

        def scoped_block():
            if name.lower() == 'c':
                assert i == 0, 'FlowNetS must be used for refinement networks'

                with tf.variable_scope('flownet_c_features'):
                    _, conv2_a, conv3_a = flownet_c_features(im1, channel_mult=channel_mult)
                    _, conv2_b, conv3_b = flownet_c_features(im2, channel_mult=channel_mult, reuse=True)

                with tf.variable_scope('flownet_c') as scope:
                    results = flownet_c(conv3_a, conv3_b, conv2_a,
                                        full_res=full_res,
                                        pose_pred = pose_prediction,
                                        channel_mult=channel_mult)
                    flows_fw.append(results[0])
                    if pose_prediction:
                        flows2_fw.append(results[1])
                        poses_fw.append(results[2])
                    if backward_flow:
                        scope.reuse_variables()
                        results = flownet_c(conv3_b, conv3_a, conv2_b,
                                            full_res=full_res,
                                            pose_pred = pose_prediction,
                                            channel_mult=channel_mult)
                        flows_bw.append(results[0])
                        if pose_prediction:
                            flows2_bw.append(results[1])
                            poses_bw.append(results[2])

            elif name.lower() == 's':
                def _flownet_s(im1, im2, flow=None):
                    if flow is not None:
                        flow = tf.image.resize_bilinear(flow, [height, width]) * 4 * FLOW_SCALE
                        warp = image_warp(im2, flow)
                        diff = tf.abs(warp - im1)
                        if not train_all:
                            flow = tf.stop_gradient(flow)
                            warp = tf.stop_gradient(warp)
                            diff = tf.stop_gradient(diff)

                        inputs = tf.concat([im1, im2, flow, warp, diff], axis=3)
                        inputs = tf.reshape(inputs, [num_batch, height, width, 14])
                    else:
                        inputs = tf.concat([im1, im2], 3)
                    return flownet_s(inputs,
                                     full_res=full_res,
                                     pose_pred=pose_prediction,
                                     channel_mult=channel_mult)
                stacked = len(flows_fw) > 0
                with tf.variable_scope('flownet_s') as scope:
                    results = _flownet_s(im1, im2, flows_fw[-1][0] if stacked else None)
                    flows_fw.append(results[0])
                    if pose_prediction:
                        flows2_fw.append(results[1])
                        poses_fw.append(results[2])

                    if backward_flow:
                        scope.reuse_variables()
                        results = _flownet_s(im2, im1, flows_bw[-1][0]  if stacked else None)
                        flows_bw.append(results[0])
                        if pose_prediction:
                            flows2_bw.append(results[1])
                            poses_bw.append(results[2])
        if i > 0:
            scope_name = "stack_{}_flownet".format(i)
            with tf.variable_scope(scope_name):
                scoped_block()
        else:
            scoped_block()

    if backward_flow :
        return flows_fw, flows_bw, flows2_fw, flows2_bw, poses_fw, poses_bw
    return flows_fw, flows2_fw, poses_fw


def _leaky_relu(x):
    with tf.variable_scope('leaky_relu'):
        return tf.maximum(0.1 * x, x)

def _flownet_upconv(conv6_1, conv5_1, conv4_1, conv3_1, conv2, conv1=None, inputs=None,
                    channel_mult=1, full_res=False, pose_pred=False, channels=2, channels_p=10):
    m = channel_mult

    flow6 = slim.conv2d(conv6_1, channels, 3, scope='flow6',
                        activation_fn=None)
    deconv5 = slim.conv2d_transpose(conv6_1, int(512 * m), 4, stride=2,
                                   scope='deconv5')
    flow6_up5 = slim.conv2d_transpose(flow6, channels, 4, stride=2,
                                     scope='flow6_up5',
                                     activation_fn=None)
    concat5 = tf.concat([conv5_1, deconv5, flow6_up5], 1)
    flow5 = slim.conv2d(concat5, channels, 3, scope='flow5',
                       activation_fn=None)

    deconv4 = slim.conv2d_transpose(concat5, int(256 * m), 4, stride=2,
                                   scope='deconv4')
    flow5_up4 = slim.conv2d_transpose(flow5, channels, 4, stride=2,
                                     scope='flow5_up4',
                                     activation_fn=None)
    concat4 = tf.concat([conv4_1, deconv4, flow5_up4], 1)
    flow4 = slim.conv2d(concat4, channels, 3, scope='flow4',
                       activation_fn=None)

    deconv3 = slim.conv2d_transpose(concat4, int(128 * m), 4, stride=2,
                                   scope='deconv3')
    flow4_up3 = slim.conv2d_transpose(flow4, channels, 4, stride=2,
                                     scope='flow4_up3',
                                     activation_fn=None)
    concat3 = tf.concat([conv3_1, deconv3, flow4_up3], 1)
    flow3 = slim.conv2d(concat3, channels, 3, scope='flow3',
                       activation_fn=None)

    deconv2 = slim.conv2d_transpose(concat3, int(64 * m), 4, stride=2,
                                   scope='deconv2')
    flow3_up2 = slim.conv2d_transpose(flow3, channels, 4, stride=2,
                                     scope='flow3_up2',
                                     activation_fn=None)
    concat2 = tf.concat([conv2, deconv2, flow3_up2], 1)
    flow2 = slim.conv2d(concat2, channels, 3, scope='flow2',
                       activation_fn=None)

    if full_res:
        with tf.variable_scope('full_res'):
            deconv1 = slim.conv2d_transpose(concat2, int(32 * m), 4, stride=2,
                                           scope='deconv1')
            flow2_up1 = slim.conv2d_transpose(flow2, channels, 4, stride=2,
                                             scope='flow2_up1',
                                             activation_fn=None)
            concat1 = tf.concat([conv1, deconv1, flow2_up1], 1)
            flow1 = slim.conv2d(concat1, channels, 3, scope='flow1',
                                activation_fn=None)

            deconv0 = slim.conv2d_transpose(concat1, int(16 * m), 4, stride=2,
                                           scope='deconv0')
            flow1_up0 = slim.conv2d_transpose(flow1, channels, 4, stride=2,
                                             scope='flow1_up0',
                                             activation_fn=None)
            concat0 = tf.concat([inputs, deconv0, flow1_up0], 1)
            flow0 = slim.conv2d(concat0, channels, 3, scope='flow0',
                                activation_fn=None)

    if pose_pred:
        def _meshgrid(batch, height, width):
            with tf.variable_scope('_meshgrid'):
                # This should be equivalent to:
                #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
                #                         np.linspace(-1, 1, height))
                #  ones = np.ones(np.prod(x_t.shape))
                #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
                x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                                tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
                y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                                tf.ones(shape=tf.stack([1, width])))
                grid = tf.expand_dims(tf.stack(axis=0, values=[x_t, y_t]), 0)
                grid = tf.tile(grid, tf.stack([batch, 1, 1, 1]))
                return grid

        with tf.variable_scope('pose_pred'):
            if full_res:
                b, _, h, w = tf.unstack(tf.shape(flow0))
                meshgrid0 = _meshgrid(b, h, w)
                b, _, h, w = tf.unstack(tf.shape(flow1))
                meshgrid1 = _meshgrid(b, h, w)

                concat0 = tf.concat([concat0, flow0, meshgrid0], 1)
                pose0 = slim.conv2d(concat0, channels_p, 3, scope='pose0',
                                    activation_fn=None)
                concat0_p = tf.concat([pose0, meshgrid0], 1)
                flow0_2 = slim.conv2d(concat0_p, channels, 1, scope='flow0',
                                    activation_fn=None)

                cnv1 = slim.conv2d(concat0, 64, [7, 7], stride=2, scope='cnv1', activation_fn=_leaky_relu)
                concat1 = tf.concat([cnv1 ,flow1, meshgrid1], 1)
                pose1 = slim.conv2d(concat1, channels_p, 3, scope='pose1',
                                    activation_fn=None)
                concat1_p = tf.concat([pose1, meshgrid1], 1)
                flow1_2 = slim.conv2d(concat1_p, channels, 1, scope='flow1',
                                    activation_fn=None)
                cnv2 = slim.conv2d(concat1, 64, [5, 5], stride=2, scope='cnv2', activation_fn=_leaky_relu)
            b, _, h, w = tf.unstack(tf.shape(flow2))
            meshgrid2 = _meshgrid(b, h, w)
            b, _, h, w = tf.unstack(tf.shape(flow3))
            meshgrid3 = _meshgrid(b, h, w)
            b, _, h, w = tf.unstack(tf.shape(flow4))
            meshgrid4 = _meshgrid(b, h, w)
            b, _, h, w = tf.unstack(tf.shape(flow5))
            meshgrid5 = _meshgrid(b, h, w)
            b, _, h, w = tf.unstack(tf.shape(flow6))
            meshgrid6 = _meshgrid(b, h, w)

            concat2 = tf.concat([flow2, meshgrid2], 1)
            cnv2_1 = slim.conv2d(concat2, 16, [3, 3], stride=1, scope='cnv2_1', activation_fn=_leaky_relu)
            cnv2_2 = slim.conv2d(cnv2_1, 32, [3, 3], stride=1, scope='cnv2_2', activation_fn=None)
            cnv2_3 = slim.conv2d(cnv2_2, 64, [3, 3], stride=1, scope='cnv2_3', activation_fn=None)
            pose2 = slim.conv2d(cnv2_3, channels_p, 3, scope='pose2',
                                activation_fn=None)
            concat2_p = tf.concat([pose2, meshgrid2], 1)
            cnvp2_2 = slim.conv2d(concat2_p, 32, [3, 3], stride=1, scope='cnvp2_2', activation_fn=None)
            cnvp2_3 = slim.conv2d(cnvp2_2, 64, [3, 3], stride=1, scope='cnvp2_3', activation_fn=None)
            flow2_2 = slim.conv2d(cnvp2_3, channels, 1, scope='flow2',
                                activation_fn=None)

            cnv3 = slim.conv2d(cnv2_3, 16, [3, 3], stride=2, scope='cnv3', activation_fn=_leaky_relu)
            concat3 = tf.concat([cnv3, flow3, meshgrid3], 1)
            cnv3_2 = slim.conv2d(concat3, 32, [3, 3], stride=1, scope='cnv3_2', activation_fn=None)
            cnv3_3 = slim.conv2d(cnv3_2, 64, [3, 3], stride=1, scope='cnv3_3', activation_fn=None)
            pose3 = slim.conv2d(cnv3_3, channels_p, 3, scope='pose3',
                                activation_fn=None)
            concat3_p = tf.concat([pose3, meshgrid3], 1)
            cnvp3_2 = slim.conv2d(concat3_p, 32, [3, 3], stride=1, scope='cnvp3_2', activation_fn=None)
            cnvp3_3 = slim.conv2d(cnvp3_2, 64, [3, 3], stride=1, scope='cnvp3_3', activation_fn=None)
            flow3_2 = slim.conv2d(cnvp3_3, channels, 1, scope='flow3',
                                activation_fn=None)

            cnv4 = slim.conv2d(cnv3_3, 16, [3, 3], stride=2, scope='cnv4', activation_fn=_leaky_relu)
            concat4 = tf.concat([cnv4, flow4, meshgrid4], 1)
            cnv4_2 = slim.conv2d(concat4, 32, [3, 3], stride=1, scope='cnv4_2', activation_fn=None)
            cnv4_3 = slim.conv2d(cnv4_2, 64, [3, 3], stride=1, scope='cnv4_3', activation_fn=None)
            pose4 = slim.conv2d(cnv4_3, channels_p, 3, scope='pose4',
                                activation_fn=None)
            concat4_p = tf.concat([pose4, meshgrid4], 1)
            cnvp4_2 = slim.conv2d(concat4_p, 32, [3, 3], stride=1, scope='cnvp4_2', activation_fn=None)
            cnvp4_3 = slim.conv2d(cnvp4_2, 64, [3, 3], stride=1, scope='cnvp4_3', activation_fn=None)
            flow4_2 = slim.conv2d(cnvp4_3, channels, 1, scope='flow4',
                                activation_fn=None)

            cnv5 = slim.conv2d(cnv4_3, 16, [3, 3], stride=2, scope='cnv5', activation_fn=_leaky_relu)
            concat5 = tf.concat([cnv5, flow5, meshgrid5], 1)
            cnv5_2 = slim.conv2d(concat5, 32, [3, 3], stride=1, scope='cnv5_2', activation_fn=None)
            cnv5_3 = slim.conv2d(cnv5_2, 64, [3, 3], stride=1, scope='cnv5_3', activation_fn=None)
            pose5 = slim.conv2d(cnv5_3, channels_p, 3, scope='pose5',
                                activation_fn=None)
            concat5_p = tf.concat([pose5, meshgrid5], 1)
            cnvp5_2 = slim.conv2d(concat5_p, 32, [3, 3], stride=1, scope='cnvp5_2', activation_fn=None)
            cnvp5_3 = slim.conv2d(cnvp5_2, 64, [3, 3], stride=1, scope='cnvp5_3', activation_fn=None)
            flow5_2 = slim.conv2d(cnvp5_3, channels, 1, scope='flow5',
                                activation_fn=None)

            cnv6 = slim.conv2d(cnv5_3, 16, [3, 3], stride=2, scope='cnv6', activation_fn=_leaky_relu)
            concat6 = tf.concat([cnv6, flow6, meshgrid6], 1)
            cnv6_2 = slim.conv2d(concat6, 32, [3, 3], stride=1, scope='cnv6_2', activation_fn=None)
            cnv6_3 = slim.conv2d(cnv6_2, 64, [3, 3], stride=1, scope='cnv6_3', activation_fn=None)
            pose6 = slim.conv2d(cnv6_3, channels_p, 3, scope='pose6',
                                activation_fn=None)
            concat6_p = tf.concat([pose6, meshgrid6], 1)
            cnvp6_2 = slim.conv2d(concat6_p, 32, [3, 3], stride=1, scope='cnvp6_2', activation_fn=None)
            cnvp6_3 = slim.conv2d(cnvp6_2, 64, [3, 3], stride=1, scope='cnvp6_3', activation_fn=None)
            flow6_2 = slim.conv2d(cnvp6_3, channels, 1, scope='flow6',
                                activation_fn=None)

    flows = [flow2, flow3, flow4, flow5, flow6]
    if full_res:
            flows = [flow0, flow1] + flows
    if pose_pred:
        poses = [pose2, pose3, pose4, pose5, pose6]
        flows2 = [flow2_2, flow3_2, flow4_2, flow5_2, flow6_2]
        if full_res:
            poses = [pose0, pose1] + poses
            flows2 = [flow0_2, flow1_2] + poses
        return flows2, flows, poses
    return flows


def nhwc_to_nchw(tensors):
    return [tf.transpose(t, [0, 3, 1, 2]) for t in tensors]


def nchw_to_nhwc(tensors):
    return [tf.transpose(t, [0, 2, 3, 1]) for t in tensors]


def flownet_p_buf(inputs, channel_mult=1, full_res=False):
    """Given stacked inputs, returns flow predictions in decreasing resolution.

    Uses FlowNetSimple.
    """
    m = channel_mult
    inputs = nhwc_to_nchw([inputs])[0]

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        data_format='NCHW',
                        weights_regularizer=slim.l2_regularizer(0.0004),
                        weights_initializer=layers.variance_scaling_initializer(),
                        activation_fn=_leaky_relu):
        conv1 = slim.conv2d(inputs, int(64 * m), 7, stride=2, scope='conv1')
        conv2 = slim.conv2d(conv1, int(128 * m), 5, stride=2, scope='conv2')
        conv3 = slim.conv2d(conv2, int(256 * m), 5, stride=2, scope='conv3')
        conv3_1 = slim.conv2d(conv3, int(256 * m), 3, stride=1, scope='conv3_1')
        conv4 = slim.conv2d(conv3_1, int(512 * m), 3, stride=2, scope='conv4')
        conv4_1 = slim.conv2d(conv4, int(512 * m), 3, stride=1, scope='conv4_1')
        conv5 = slim.conv2d(conv4_1, int(512 * m), 3, stride=2, scope='conv5')
        conv5_1 = slim.conv2d(conv5, int(512 * m), 3, stride=1, scope='conv5_1')
        conv6 = slim.conv2d(conv5_1, int(1024 * m), 3, stride=2, scope='conv6')
        conv6_1 = slim.conv2d(conv6, int(1024 * m), 3, stride=1, scope='conv6_1')

        flows, poses = _flownet_p_upconv(conv6_1, conv5_1, conv4_1, conv3_1, conv2, conv1, inputs,
                              channel_mult=channel_mult, full_res=full_res)
        return nchw_to_nhwc(flows), nchw_to_nhwc(poses)

def _flownet_p_upconv(conv6_1, conv5_1, conv4_1, conv3_1, conv2, conv1=None, inputs=None,
                    channel_mult=1, full_res=False, channels=2, channels_p=8):
    m = channel_mult

    pose6 = slim.conv2d(conv6_1, channels_p, 3, scope='pose6',
                        activation_fn=None)
    flow6 = slim.conv2d(pose6, channels, 1, scope='flow6',
                        activation_fn=None)
    deconv5 = slim.conv2d_transpose(conv6_1, int(512 * m), 4, stride=2,
                                   scope='deconv5')
    pose6_up5 = slim.conv2d_transpose(pose6, channels_p, 4, stride=2,
                                     scope='pose6_up5',
                                     activation_fn=None)
    flow6_up5 = slim.conv2d_transpose(flow6, channels, 4, stride=2,
                                     scope='flow6_up5',
                                     activation_fn=None)
    concat5 = tf.concat([conv5_1, deconv5, pose6_up5, flow6_up5], 1)

    pose5 = slim.conv2d(concat5, channels_p, 3, scope='pose5',
                       activation_fn=None)
    flow5 = slim.conv2d(pose5, channels, 1, scope='flow5',
                        activation_fn=None)
    deconv4 = slim.conv2d_transpose(concat5, int(256 * m), 4, stride=2,
                                   scope='deconv4')
    pose5_up4 = slim.conv2d_transpose(pose5, channels_p, 4, stride=2,
                                     scope='pose5_up4',
                                     activation_fn=None)
    flow5_up4 = slim.conv2d_transpose(flow5, channels, 4, stride=2,
                                     scope='flow5_up4',
                                     activation_fn=None)
    concat4 = tf.concat([conv4_1, deconv4, pose5_up4, flow5_up4], 1)

    pose4 = slim.conv2d(concat4, channels_p, 3, scope='pose4',
                       activation_fn=None)
    flow4 = slim.conv2d(pose4, channels, 1, scope='flow4',
                        activation_fn=None)
    deconv3 = slim.conv2d_transpose(concat4, int(128 * m), 4, stride=2,
                                   scope='deconv3')
    pose4_up3 = slim.conv2d_transpose(pose4, channels_p, 4, stride=2,
                                     scope='pose4_up3',
                                     activation_fn=None)
    flow4_up3 = slim.conv2d_transpose(flow4, channels, 4, stride=2,
                                     scope='flow4_up3',
                                     activation_fn=None)
    concat3 = tf.concat([conv3_1, deconv3, pose4_up3, flow4_up3], 1)

    pose3 = slim.conv2d(concat3, channels_p, 3, scope='pose3',
                       activation_fn=None)
    flow3 = slim.conv2d(pose3, channels, 1, scope='flow3',
                        activation_fn=None)

    deconv2 = slim.conv2d_transpose(concat3, int(64 * m), 4, stride=2,
                                   scope='deconv2')
    pose3_up2 = slim.conv2d_transpose(pose3, channels_p, 4, stride=2,
                                     scope='pose3_up2',
                                     activation_fn=None)
    flow3_up2 = slim.conv2d_transpose(flow3, channels, 4, stride=2,
                                     scope='flow3_up2',
                                     activation_fn=None)
    concat2 = tf.concat([conv2, deconv2, pose3_up2, flow3_up2], 1)
    pose2 = slim.conv2d(concat2, channels_p, 3, scope='pose2',
                       activation_fn=None)
    flow2 = slim.conv2d(pose2, channels, 1, scope='flow2',
                        activation_fn=None)

    flows = [flow2, flow3, flow4, flow5, flow6]
    poses = [pose2, pose3, pose4, pose5, pose6]

    if full_res:
        with tf.variable_scope('full_res'):
            deconv1 = slim.conv2d_transpose(concat2, int(32 * m), 4, stride=2,
                                           scope='deconv1')
            pose2_up1 = slim.conv2d_transpose(pose2, channels_p, 4, stride=2,
                                             scope='pose2_up1',
                                             activation_fn=None)
            flow2_up1 = slim.conv2d_transpose(flow2, channels, 4, stride=2,
                                             scope='flow2_up1',
                                             activation_fn=None)
            concat1 = tf.concat([conv1, deconv1, pose2_up1, flow2_up1], 1)
            pose1 = slim.conv2d(concat1, channels_p, 3, scope='pose1',
                                        activation_fn=None)
            flow1 = slim.conv2d(pose1, channels, 1, scope='flow1',
                                activation_fn=None)

            deconv0 = slim.conv2d_transpose(concat1, int(16 * m), 4, stride=2,
                                           scope='deconv0')
            pose1_up0 = slim.conv2d_transpose(pose1, channels_p, 4, stride=2,
                                             scope='pose1_up0',
                                             activation_fn=None)
            flow1_up0 = slim.conv2d_transpose(flow1, channels, 4, stride=2,
                                             scope='flow1_up0',
                                             activation_fn=None)
            concat0 = tf.concat([inputs, deconv0, pose1_up0, flow1_up0], 1)
            pose0 = slim.conv2d(concat0, channels_p, 3, scope='pose0',
                                activation_fn=None)
            flow0 = slim.conv2d(pose0, channels, 1, scope='flow0',
                                activation_fn=None)

            flows = [flow0, flow1] + flows
            poses = [pose0, pose1] + poses

    return flows, poses

def flownet_s(inputs, channel_mult=1, full_res=False, pose_pred=False):
    """Given stacked inputs, returns flow predictions in decreasing resolution.

    Uses FlowNetSimple.
    """
    m = channel_mult
    inputs = nhwc_to_nchw([inputs])[0]

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        data_format='NCHW',
                        weights_regularizer=slim.l2_regularizer(0.0004),
                        weights_initializer=layers.variance_scaling_initializer(),
                        activation_fn=_leaky_relu):
        conv1 = slim.conv2d(inputs, int(64 * m), 7, stride=2, scope='conv1')
        conv2 = slim.conv2d(conv1, int(128 * m), 5, stride=2, scope='conv2')
        conv3 = slim.conv2d(conv2, int(256 * m), 5, stride=2, scope='conv3')
        conv3_1 = slim.conv2d(conv3, int(256 * m), 3, stride=1, scope='conv3_1')
        conv4 = slim.conv2d(conv3_1, int(512 * m), 3, stride=2, scope='conv4')
        conv4_1 = slim.conv2d(conv4, int(512 * m), 3, stride=1, scope='conv4_1')
        conv5 = slim.conv2d(conv4_1, int(512 * m), 3, stride=2, scope='conv5')
        conv5_1 = slim.conv2d(conv5, int(512 * m), 3, stride=1, scope='conv5_1')
        conv6 = slim.conv2d(conv5_1, int(1024 * m), 3, stride=2, scope='conv6')
        conv6_1 = slim.conv2d(conv6, int(1024 * m), 3, stride=1, scope='conv6_1')

        if pose_pred:
            flows1, flows2, poses = _flownet_upconv(conv6_1, conv5_1, conv4_1, conv3_1, conv2, conv1, inputs,
                                  channel_mult=channel_mult, full_res=full_res, pose_pred=pose_pred)
            return [nchw_to_nhwc(flows1), nchw_to_nhwc(flows2), nchw_to_nhwc(poses)]
        else:
            res = _flownet_upconv(conv6_1, conv5_1, conv4_1, conv3_1, conv2, conv1, inputs,
                                  channel_mult=channel_mult, full_res=full_res)
            return [nchw_to_nhwc(res)]


def flownet_c_features(im, channel_mult=1, reuse=None):
    m = channel_mult
    im = nhwc_to_nchw([im])[0]
    with slim.arg_scope([slim.conv2d],
                        data_format='NCHW',
                        weights_regularizer=slim.l2_regularizer(0.0004),
                        weights_initializer=layers.variance_scaling_initializer(),
                        activation_fn=_leaky_relu):
        conv1 = slim.conv2d(im, int(64 * m), 7, stride=2, scope='conv1', reuse=reuse)
        conv2 = slim.conv2d(conv1, int(128 * m), 5, stride=2, scope='conv2', reuse=reuse)
        conv3 = slim.conv2d(conv2, int(256 * m), 5, stride=2, scope='conv3', reuse=reuse)
        return conv1, conv2, conv3


def flownet_c(conv3_a, conv3_b, conv2_a, channel_mult=1, full_res=False, pose_pred=False):
    """Given two images, returns flow predictions in decreasing resolution.

    Uses FlowNetCorr.
    """
    m = channel_mult

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                        data_format='NCHW',
                        weights_regularizer=slim.l2_regularizer(0.0004),
                        weights_initializer=layers.variance_scaling_initializer(),
                        activation_fn=_leaky_relu):
        corr = correlation(conv3_a, conv3_b,
                           pad=20, kernel_size=1, max_displacement=20, stride_1=1, stride_2=2)

        conv_redir = slim.conv2d(conv3_a, int(32 * m), 1, stride=1, scope='conv_redir')

        conv3_1 = slim.conv2d(tf.concat([conv_redir, corr], 1), int(256 * m), 3,
                              stride=1, scope='conv3_1')
        conv4 = slim.conv2d(conv3_1, int(512 * m), 3, stride=2, scope='conv4')
        conv4_1 = slim.conv2d(conv4, int(512 * m), 3, stride=1, scope='conv4_1')
        conv5 = slim.conv2d(conv4_1, int(512 * m), 3, stride=2, scope='conv5')
        conv5_1 = slim.conv2d(conv5, int(512 * m), 3, stride=1, scope='conv5_1')
        conv6 = slim.conv2d(conv5_1, int(1024 * m), 3, stride=2, scope='conv6')
        conv6_1 = slim.conv2d(conv6, int(1024 * m), 3, stride=1, scope='conv6_1')

        if pose_pred:
            flows1, flows2, poses = _flownet_upconv(conv6_1, conv5_1, conv4_1, conv3_1, conv2_a,
                                  channel_mult=channel_mult, full_res=full_res, pose_pred=pose_pred)
            return [nchw_to_nhwc(flows1), nchw_to_nhwc(flows2), nchw_to_nhwc(poses)]
        else:
            flows = _flownet_upconv(conv6_1, conv5_1, conv4_1, conv3_1, conv2_a,
                                  channel_mult=channel_mult, full_res=full_res, pose_pred=pose_pred)
            return [nchw_to_nhwc(flows)]

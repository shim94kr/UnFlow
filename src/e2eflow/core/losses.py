import tensorflow as tf
import numpy as np
from tensorflow.contrib.distributions import Normal

from ..ops import backward_warp, forward_warp
from .image_warp import image_warp
from .util import make_grid, posegrid_vec2mat


DISOCC_THRESH = 0.8

def length_sq(x):
    return tf.reduce_sum(tf.square(x), 3, keep_dims=True)


def compute_losses(im1, im2, flow_fw, flow_bw,
                   pose_fw=None, pose_bw=None, intrinsic1=None, intrinsic2=None,
                   flow2_fw=None, flow2_bw=None,
                   border_mask=None,
                   mask_occlusion='',
                   data_max_distance=1):
    losses = {}

    im2_warped = image_warp(im2, flow_fw)
    im1_warped = image_warp(im1, flow_bw)

    im_diff_fw = im1 - im2_warped
    im_diff_bw = im2 - im1_warped

    disocc_fw = tf.cast(forward_warp(flow_fw) < DISOCC_THRESH, tf.float32)
    disocc_bw = tf.cast(forward_warp(flow_bw) < DISOCC_THRESH, tf.float32)

    if border_mask is None:
        mask_fw = create_outgoing_mask(flow_fw)
        mask_bw = create_outgoing_mask(flow_bw)
    else:
        mask_fw = border_mask
        mask_bw = border_mask

    mag_sq = length_sq(flow_fw) + length_sq(flow_bw)
    flow_bw_warped = image_warp(flow_bw, flow_fw)
    flow_fw_warped = image_warp(flow_fw, flow_bw)
    flow_diff_fw = flow_fw + flow_bw_warped
    flow_diff_bw = flow_bw + flow_fw_warped
    occ_thresh =  0.01 * mag_sq + 0.5
    fb_occ_fw = tf.cast(length_sq(flow_diff_fw) > occ_thresh, tf.float32)
    fb_occ_bw = tf.cast(length_sq(flow_diff_bw) > occ_thresh, tf.float32)

    if mask_occlusion == 'fb':
        mask_fw *= (1 - fb_occ_fw)
        mask_bw *= (1 - fb_occ_bw)
    elif mask_occlusion == 'disocc':
        mask_fw *= (1 - disocc_bw)
        mask_bw *= (1 - disocc_fw)
    elif mask_occlusion == 'both':
        mask_fw *= ((2 - disocc_bw - fb_occ_fw)/2)
        mask_bw *= ((2 - disocc_fw - fb_occ_bw)/2)

    """
    zeros = tf.zeros_like(border_mask)
    mask_dyn_fw = tf.maximum(border_mask * (1 - disocc_fw - fb_occ_fw), zeros)
    mask_dyn_bw = tf.maximum(border_mask * (1 - disocc_bw - fb_occ_bw), zeros)
    """
    occ_fw = 1 - mask_fw
    occ_bw = 1 - mask_bw

    losses['sym'] = (charbonnier_loss(occ_fw - disocc_bw) +
                     charbonnier_loss(occ_bw - disocc_fw))

    losses['occ'] = (charbonnier_loss(occ_fw) +
                     charbonnier_loss(occ_bw))

    losses['photo'] =  (photometric_loss(im_diff_fw, mask_fw) +
                        photometric_loss(im_diff_bw, mask_bw))

    losses['grad'] = (gradient_loss(im1, im2_warped, mask_fw) +
                      gradient_loss(im2, im1_warped, mask_bw))

    losses['smooth_1st'] = (smoothness_loss(flow_fw, im1) +
                            smoothness_loss(flow_bw, im2))

    losses['smooth_2nd'] = (second_order_loss(flow_fw) +
                            second_order_loss(flow_bw))

    losses['fb'] = (charbonnier_loss(flow_diff_fw, mask_fw) +
                    charbonnier_loss(flow_diff_bw, mask_bw))


    losses['ternary'] = (ternary_loss(im1, im2_warped, mask_fw,
                                      max_distance=data_max_distance) +
                         ternary_loss(im2, im1_warped, mask_bw,
                                      max_distance=data_max_distance))

    if pose_fw is not None:
        im2_warped = image_warp(im2, flow2_fw)
        im1_warped = image_warp(im1, flow2_bw)
        flow2_bw_warped = image_warp(flow2_bw, flow2_fw)
        flow2_fw_warped = image_warp(flow2_fw, flow2_bw)
        flow2_diff_fw = flow2_fw + flow2_bw_warped
        flow2_diff_bw = flow2_bw + flow2_fw_warped
        #pose_bw_warped = image_warp(pose_bw, flow_fw)
        #pose_fw_warped = image_warp(pose_fw, flow_bw)
        losses['ternary'] += (ternary_loss(im1, im2_warped, mask_fw,
                                          max_distance=data_max_distance) +
                             ternary_loss(im2, im1_warped, mask_bw,
                                              max_distance=data_max_distance))
        losses['smooth_2nd'] += (second_order_loss(flow2_fw) +
                                second_order_loss(flow2_bw))
        losses['fb'] += (charbonnier_loss(flow2_diff_fw, mask_fw) +
                        charbonnier_loss(flow2_diff_bw, mask_bw))

        rot1, trans1, sig1, pose_fw = posegrid_vec2mat(pose_fw)
        rot2, trans2, sig2, pose_bw = posegrid_vec2mat(pose_bw)
        #rot1_wp, trans1_wp, _, _ = posegrid_vec2mat(pose_bw_warped)
        #rot2_wp, trans2_wp, _, _ = posegrid_vec2mat(pose_fw_warped)
        pose_fw_all = tf.concat([pose_fw, sig1], axis=3)
        pose_bw_all = tf.concat([pose_bw, sig2], axis=3)
        losses['epipolar'] = (epipolar_loss(flow_fw, rot1, trans1, intrinsic1, intrinsic2, mask_fw) + \
                             epipolar_loss(flow_bw, rot2, trans2, intrinsic1, intrinsic2, mask_bw))
        losses['smooth_pose_2nd'] = (pose_second_order_loss(pose_fw_all) + pose_second_order_loss(pose_bw_all))
        losses['pose_scale'] = 0 #charbonnier_loss(sig1, mask_fw) + charbonnier_loss(sig2, mask_fw)
        losses['sym_pose'] = 0#sym_pose_loss(rot1, trans1, rot1_wp, trans1_wp) + sym_pose_loss(rot2, trans2, rot2_wp, trans2_wp)
        return losses, pose_fw_all, pose_bw_all
    else :
        losses['epipolar'] = 0
        losses['smooth_pose_2nd'] = 0
        losses['pose_scale'] = 0
        losses['sym_pose'] = 0
    
    return losses

def sym_pose_loss(rot, trans, rot_wp, trans_wp):
    batch_size, H, W, _, _ = tf.unstack(tf.shape(rot))
    identity_mat = tf.eye(3, batch_shape=[batch_size])
    identity_mat = tf.tile(tf.expand_dims(tf.expand_dims(identity_mat, 1), 1), [1, H, W, 1, 1])
    rot_res = tf.matmul(rot, rot_wp) - identity_mat

    trans = tf.expand_dims(trans, -1)
    trans_wp = tf.expand_dims(trans_wp, -1)
    trans_res = tf.matmul(rot_wp, trans) - trans_wp
    sym_rot_error = tf.pow(tf.square(rot_res) + tf.square(0.001), 0.45)
    sym_trans_error = tf.pow(tf.square(trans_res) + tf.square(0.001), 0.45)
    return tf.reduce_mean(sym_rot_error) + tf.reduce_mean(sym_trans_error)

def epipolar_loss(flow, rot, trans, intrinsic1, intrinsic2, mask, forward=True):
    batch_size, H, W, _ = tf.unstack(tf.shape(flow))
    grid_tgt = make_grid(batch_size, H, W)
    grid_src_from_tgt = grid_tgt[:, :, :, 0:2] + flow[:, :, :, 0:2]

    grid_tgt = tf.expand_dims(grid_tgt, -1)

    trans = tf.tile(tf.expand_dims(trans, 3), [1, 1, 1, 3, 1]) # B * H * W * 3 * 3
    essential_matrix  = tf.cross(rot, trans) # B * H * W * 3 * 3
    inv_intrinsic1 = tf.tile(tf.expand_dims(tf.expand_dims(tf.matrix_inverse(intrinsic1), 1), 1), [1, H, W, 1, 1]) # B * H * W * 3 * 3
    inv_intrinsic2 = tf.tile(tf.expand_dims(tf.expand_dims(tf.matrix_inverse(intrinsic2), 1), 1), [1, H, W, 1, 1]) # B * H * W * 3 * 3
    fundamental_matrix = tf.matmul(tf.matmul(tf.transpose(inv_intrinsic2, [0, 1, 2, 4, 3]), essential_matrix), inv_intrinsic1) # B * H * W * 3 * 3

    grid_src_cct1 = tf.concat([grid_src_from_tgt, tf.ones([batch_size, H, W, 1])], axis = 3)
    grid_src = tf.reshape(grid_src_cct1, [batch_size, H, W, 1, 3]) # B * H * W * 1 * 3
    epipolar_error = tf.matmul(tf.matmul(grid_src, fundamental_matrix), grid_tgt) # B * H * W * 1 * 1
    epipolar_error = tf.squeeze(epipolar_error, axis=[-1]) # B * H * W * 1

    return tf.reduce_mean(tf.abs(epipolar_error * mask))


def ternary_loss(im1, im2_warped, mask, max_distance=1):
    patch_size = 2 * max_distance + 1
    with tf.variable_scope('ternary_loss'):
        def _ternary_transform(image):
            intensities = tf.image.rgb_to_grayscale(image) * 255
            #patches = tf.extract_image_patches( # fix rows_in is None
            #    intensities,
            #    ksizes=[1, patch_size, patch_size, 1],
            #    strides=[1, 1, 1, 1],
            #    rates=[1, 1, 1, 1],
            #    padding='SAME')
            out_channels = patch_size * patch_size
            w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
            weights =  tf.constant(w, dtype=tf.float32)
            patches = tf.nn.conv2d(intensities, weights, strides=[1, 1, 1, 1], padding='SAME')

            transf = patches - intensities
            transf_norm = transf / tf.sqrt(0.81 + tf.square(transf))
            return transf_norm

        def _hamming_distance(t1, t2):
            dist = tf.square(t1 - t2)
            dist_norm = dist / (0.1 + dist)
            dist_sum = tf.reduce_sum(dist_norm, 3, keep_dims=True)
            return dist_sum

        t1 = _ternary_transform(im1)
        t2 = _ternary_transform(im2_warped)
        dist = _hamming_distance(t1, t2)

        transform_mask = create_mask(mask, [[max_distance, max_distance],
                                            [max_distance, max_distance]])
        return charbonnier_loss(dist, mask * transform_mask)


def occlusion(flow_fw, flow_bw):
    mag_sq = length_sq(flow_fw) + length_sq(flow_bw)
    flow_bw_warped = image_warp(flow_bw, flow_fw)
    flow_fw_warped = image_warp(flow_fw, flow_bw)
    flow_diff_fw = flow_fw + flow_bw_warped
    flow_diff_bw = flow_bw + flow_fw_warped
    occ_thresh =  0.01 * mag_sq + 0.5
    occ_fw = tf.cast(length_sq(flow_diff_fw) > occ_thresh, tf.float32)
    occ_bw = tf.cast(length_sq(flow_diff_bw) > occ_thresh, tf.float32)
    return occ_fw, occ_bw


#def disocclusion(div):
#    """Creates binary disocclusion map based on flow divergence."""
#    return tf.round(norm(tf.maximum(0.0, div), 0.3))


#def occlusion(im_diff, div):
#    """Creates occlusion map based on warping error & flow divergence."""
#    gray_diff = tf.image.rgb_to_grayscale(im_diff)
#    return 1 - norm(gray_diff, 20.0 / 255) * norm(tf.minimum(0.0, div), 0.3)


def divergence(flow):
    with tf.variable_scope('divergence'):
        filter_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] # sobel filter
        filter_y = np.transpose(filter_x)
        weight_array_x = np.zeros([3, 3, 1, 1])
        weight_array_x[:, :, 0, 0] = filter_x
        weights_x = tf.constant(weight_array_x, dtype=tf.float32)
        weight_array_y = np.zeros([3, 3, 1, 1])
        weight_array_y[:, :, 0, 0] = filter_y
        weights_y = tf.constant(weight_array_y, dtype=tf.float32)
        flow_u, flow_v = tf.split(axis=3, num_or_size_splits=2, value=flow)
        grad_x = conv2d(flow_u, weights_x)
        grad_y = conv2d(flow_v, weights_y)
        div = tf.reduce_sum(tf.concat(axis=3, values=[grad_x, grad_y]), 3, keep_dims=True)
        return div


def norm(x, sigma):
    """Gaussian decay.
    Result is 1.0 for x = 0 and decays towards 0 for |x > sigma.
    """
    dist = Normal(0.0, sigma)
    return dist.pdf(x) / dist.pdf(0.0)


def diffusion_loss(flow, im, occ):
    """Forces diffusion weighted by motion, intensity and occlusion label similarity.
    Inspired by Bilateral Flow Filtering.
    """
    def neighbor_diff(x, num_in=1):
        weights = np.zeros([3, 3, num_in, 8 * num_in])
        out_channel = 0
        for c in range(num_in): # over input channels
            for n in [0, 1, 2, 3, 5, 6, 7, 8]: # over neighbors
                weights[1, 1, c, out_channel] = 1
                weights[n // 3, n % 3, c, out_channel] = -1
                out_channel += 1
        weights = tf.constant(weights, dtype=tf.float32)
        return conv2d(x, weights)

    # Create 8 channel (one per neighbor) differences
    occ_diff = neighbor_diff(occ)
    flow_diff_u, flow_diff_v = tf.split(axis=3, num_or_size_splits=2, value=neighbor_diff(flow, 2))
    flow_diff = tf.sqrt(tf.square(flow_diff_u) + tf.square(flow_diff_v))
    intensity_diff = tf.abs(neighbor_diff(tf.image.rgb_to_grayscale(im)))

    diff = norm(intensity_diff, 7.5 / 255) * norm(flow_diff, 0.5) * occ_diff * flow_diff
    return charbonnier_loss(diff)


def photometric_loss(im_diff, mask):
    return charbonnier_loss(im_diff, mask, beta=255)


def conv2d(x, weights):
    return tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')

def _gradient_delta(im1, im2_warped):
    with tf.variable_scope('gradient_delta'):
        filter_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] # sobel filter
        filter_y = np.transpose(filter_x)
        weight_array = np.zeros([3, 3, 3, 6])
        for c in range(3):
            weight_array[:, :, c, 2 * c] = filter_x
            weight_array[:, :, c, 2 * c + 1] = filter_y
        weights = tf.constant(weight_array, dtype=tf.float32)

        im1_grad = conv2d(im1, weights)
        im2_warped_grad = conv2d(im2_warped, weights)
        diff = im1_grad - im2_warped_grad
        return diff


def gradient_loss(im1, im2_warped, mask):
    with tf.variable_scope('gradient_loss'):
        mask_x = create_mask(im1, [[0, 0], [1, 1]])
        mask_y = create_mask(im1, [[1, 1], [0, 0]])
        gradient_mask = tf.tile(tf.concat(axis=3, values=[mask_x, mask_y]), [1, 1, 1, 3])
        diff = _gradient_delta(im1, im2_warped)
        return charbonnier_loss(diff, mask * gradient_mask)

def _smoothness_deltas(flow, img=None):
    with tf.variable_scope('smoothness_delta'):
        mask_x = create_mask(flow, [[0, 0], [0, 1]])
        mask_y = create_mask(flow, [[0, 1], [0, 0]])
        mask = tf.concat(axis=3, values=[mask_x, mask_y])

        filter_x = [[0, 0, 0], [0, 1, -1], [0, 0, 0]]
        filter_y = [[0, 0, 0], [0, 1, 0], [0, -1, 0]]
        weight_array = np.ones([3, 3, 1, 2])
        weight_array[:, :, 0, 0] = filter_x
        weight_array[:, :, 0, 1] = filter_y
        weights = tf.constant(weight_array, dtype=tf.float32)

        flow_u, flow_v = tf.split(axis=3, num_or_size_splits=2, value=flow)
        delta_u = conv2d(flow_u, weights)
        delta_v = conv2d(flow_v, weights)

        img_r, img_g, img_b = tf.split(axis=3, num_or_size_splits=3, value=img)
        delta_Ir = conv2d(img_r, weights)
        delta_Ig = conv2d(img_g, weights)
        delta_Ib = conv2d(img_b, weights)
        delta_I = tf.concat([delta_Ir, delta_Ig, delta_Ib], axis=3)
        return tf.abs(delta_u) + tf.abs(delta_v), tf.norm(delta_I, axis=3, keep_dims=True), mask

def smoothness_loss(flow, img):
    with tf.variable_scope('smoothness_loss'):
        delta_m, delta_I, mask = _smoothness_deltas(flow, img)
        return tf.reduce_mean(tf.exp(-tf.pow(delta_I , 2.0)*0.1) * delta_m * mask)

def _smoothness_deltas_orig(flow):
    with tf.variable_scope('smoothness_delta'):
        mask_x = create_mask(flow, [[0, 0], [0, 1]])
        mask_y = create_mask(flow, [[0, 1], [0, 0]])
        mask = tf.concat(axis=3, values=[mask_x, mask_y])

        filter_x = [[0, 0, 0], [0, 1, -1], [0, 0, 0]]
        filter_y = [[0, 0, 0], [0, 1, 0], [0, -1, 0]]
        weight_array = np.ones([3, 3, 1, 2])
        weight_array[:, :, 0, 0] = filter_x
        weight_array[:, :, 0, 1] = filter_y
        weights = tf.constant(weight_array, dtype=tf.float32)

        flow_u, flow_v = tf.split(axis=3, num_or_size_splits=2, value=flow)
        delta_u = conv2d(flow_u, weights)
        delta_v = conv2d(flow_v, weights)
        return delta_u, delta_v, mask

def smoothness_loss_orig(flow):
    with tf.variable_scope('smoothness_loss'):
        delta_u, delta_v, mask = _smoothness_deltas_orig(flow)
        loss_u = charbonnier_loss(delta_u, mask)
        loss_v = charbonnier_loss(delta_v, mask)
        return loss_u + loss_v

def _pose_second_order_deltas(pose):
    with tf.variable_scope('_pose_second_order_deltas'):
        mask_x = create_mask(pose, [[0, 0], [1, 1]])
        mask_y = create_mask(pose, [[1, 1], [0, 0]])
        mask_diag = create_mask(pose, [[1, 1], [1, 1]])
        mask = tf.concat(axis=3, values=[mask_x, mask_y, mask_diag, mask_diag])

        filter_x = [[0, 0, 0],
                    [1, -2, 1],
                    [0, 0, 0]]
        filter_y = [[0, 1, 0],
                    [0, -2, 0],
                    [0, 1, 0]]
        filter_diag1 = [[1, 0, 0],
                        [0, -2, 0],
                        [0, 0, 1]]
        filter_diag2 = [[0, 0, 1],
                        [0, -2, 0],
                        [1, 0, 0]]
        weight_array = np.ones([3, 3, 1, 4])
        weight_array[:, :, 0, 0] = filter_x
        weight_array[:, :, 0, 1] = filter_y
        weight_array[:, :, 0, 2] = filter_diag1
        weight_array[:, :, 0, 3] = filter_diag2
        weights = tf.constant(weight_array, dtype=tf.float32)

        pose_tx, pose_ty, pose_tz, pose_rx, pose_ry, pose_rz, sig_tx, sig_ty, sig_tz = tf.split(axis=3, num_or_size_splits=9, value=pose)
        delta_tx = conv2d(pose_tx, weights)
        delta_ty = conv2d(pose_ty, weights)
        delta_tz = conv2d(pose_tz, weights)
        delta_rx = conv2d(pose_rx, weights)
        delta_ry = conv2d(pose_ry, weights)
        delta_rz = conv2d(pose_rz, weights)
        delta_sigtx = conv2d(sig_tx, weights)
        delta_sigty = conv2d(sig_ty, weights)
        delta_sigtz = conv2d(sig_tz, weights)
        return delta_tx, delta_ty, delta_tz, delta_rx, delta_ry, delta_rz, delta_sigtx, delta_sigty, delta_sigtz, mask


def pose_second_order_loss(pose):
    with tf.variable_scope('pose_second_order_loss'):
        delta_tx, delta_ty, delta_tz, delta_rx, delta_ry, delta_rz, delta_sigtx, delta_sigty, delta_sigtz, mask = _pose_second_order_deltas(pose)
        loss_tx = charbonnier_loss(delta_tx, mask)
        loss_ty = charbonnier_loss(delta_ty, mask)
        loss_tz = charbonnier_loss(delta_tz, mask)
        loss_rx = charbonnier_loss(delta_rx, mask)
        loss_ry = charbonnier_loss(delta_ry, mask)
        loss_rz = charbonnier_loss(delta_rz, mask)
        loss_sigtx = charbonnier_loss(delta_sigtx, mask)
        loss_sigty = charbonnier_loss(delta_sigty, mask)
        loss_sigtz = charbonnier_loss(delta_sigtz, mask)
        return loss_tx + loss_ty + loss_tz + loss_rx + loss_ry + loss_rz + loss_sigtx + loss_sigty + loss_sigtz


def _second_order_deltas(flow):
    with tf.variable_scope('_second_order_deltas'):
        mask_x = create_mask(flow, [[0, 0], [1, 1]])
        mask_y = create_mask(flow, [[1, 1], [0, 0]])
        mask_diag = create_mask(flow, [[1, 1], [1, 1]])
        mask = tf.concat(axis=3, values=[mask_x, mask_y, mask_diag, mask_diag])

        filter_x = [[0, 0, 0],
                    [1, -2, 1],
                    [0, 0, 0]]
        filter_y = [[0, 1, 0],
                    [0, -2, 0],
                    [0, 1, 0]]
        filter_diag1 = [[1, 0, 0],
                        [0, -2, 0],
                        [0, 0, 1]]
        filter_diag2 = [[0, 0, 1],
                        [0, -2, 0],
                        [1, 0, 0]]
        weight_array = np.ones([3, 3, 1, 4])
        weight_array[:, :, 0, 0] = filter_x
        weight_array[:, :, 0, 1] = filter_y
        weight_array[:, :, 0, 2] = filter_diag1
        weight_array[:, :, 0, 3] = filter_diag2
        weights = tf.constant(weight_array, dtype=tf.float32)

        flow_u, flow_v = tf.split(axis=3, num_or_size_splits=2, value=flow)
        delta_u = conv2d(flow_u, weights)
        delta_v = conv2d(flow_v, weights)
        return delta_u, delta_v, mask


def second_order_loss(flow):
    with tf.variable_scope('second_order_loss'):
        delta_u, delta_v, mask = _second_order_deltas(flow)
        loss_u = charbonnier_loss(delta_u, mask)
        loss_v = charbonnier_loss(delta_v, mask)
        return loss_u + loss_v


def charbonnier_loss(x, mask=None, truncate=None, alpha=0.45, beta=1.0, epsilon=0.001):
    """Compute the generalized charbonnier loss of the difference tensor x.
    All positions where mask == 0 are not taken into account.

    Args:
        x: a tensor of shape [num_batch, height, width, channels].
        mask: a mask of shape [num_batch, height, width, mask_channels],
            where mask channels must be either 1 or the same number as
            the number of channels of x. Entries should be 0 or 1.
    Returns:
        loss as tf.float32
    """
    with tf.variable_scope('charbonnier_loss'):
        batch, height, width, channels = tf.unstack(tf.shape(x))
        normalization = tf.cast(batch * height * width * channels, tf.float32)

        error = tf.pow(tf.square(x * beta) + tf.square(epsilon), alpha)

        if mask is not None:
            error = tf.multiply(mask, error)

        if truncate is not None:
            error = tf.minimum(error, truncate)

        return tf.reduce_sum(error) / normalization


def create_mask(tensor, paddings):
    with tf.variable_scope('create_mask'):
        shape = tf.shape(tensor)
        inner_width = shape[1] - (paddings[0][0] + paddings[0][1])
        inner_height = shape[2] - (paddings[1][0] + paddings[1][1])
        inner = tf.ones([inner_width, inner_height])

        mask2d = tf.pad(inner, paddings)
        mask3d = tf.tile(tf.expand_dims(mask2d, 0), [shape[0], 1, 1])
        mask4d = tf.expand_dims(mask3d, 3)
        return tf.stop_gradient(mask4d)


def create_border_mask(tensor, border_ratio=0.1):
    with tf.variable_scope('create_border_mask'):
        num_batch, height, width, _ = tf.unstack(tf.shape(tensor))
        min_dim = tf.cast(tf.minimum(height, width), 'float32')
        sz = tf.cast(tf.ceil(min_dim * border_ratio), 'int32')
        border_mask = create_mask(tensor, [[sz, sz], [sz, sz]])
        return tf.stop_gradient(border_mask)


def create_outgoing_mask(flow):
    """Computes a mask that is zero at all positions where the flow
    would carry a pixel over the image boundary."""
    with tf.variable_scope('create_outgoing_mask'):
        num_batch, height, width, _ = tf.unstack(tf.shape(flow))

        grid_x = tf.reshape(tf.range(width), [1, 1, width])
        grid_x = tf.tile(grid_x, [num_batch, height, 1])
        grid_y = tf.reshape(tf.range(height), [1, height, 1])
        grid_y = tf.tile(grid_y, [num_batch, 1, width])

        flow_u, flow_v = tf.unstack(flow, 2, 3)
        pos_x = tf.cast(grid_x, dtype=tf.float32) + flow_u
        pos_y = tf.cast(grid_y, dtype=tf.float32) + flow_v
        inside_x = tf.logical_and(pos_x <= tf.cast(width - 1, tf.float32),
                                  pos_x >=  0.0)
        inside_y = tf.logical_and(pos_y <= tf.cast(height - 1, tf.float32),
                                  pos_y >=  0.0)
        inside = tf.logical_and(inside_x, inside_y)
        return tf.expand_dims(tf.cast(inside, tf.float32), 3)
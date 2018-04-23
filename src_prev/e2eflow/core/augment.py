import numpy as np
import tensorflow as tf

from .spatial_transformer import transformer


def random_affine(tensors, *,
                  max_translation_x=0.0, max_translation_y=0.0,
                  max_rotation=0.0, min_scale=1.0, max_scale=1.0,
                  horizontal_flipping=False, intrinsics=None):
    """Applies geometric augmentations to a list of tensors.

    Each element in the list is augmented in the same way.
    For all elements, num_batch must be equal while height, width and channels
    may differ.
    """
    def _deg2rad(deg):
        return (deg * np.pi) / 180.0

    with tf.variable_scope('random_affine'):
        num_batch = tf.shape(tensors[0])[0]

        zero = tf.zeros([num_batch])
        one = tf.ones([num_batch])

        tx = tf.random_uniform([num_batch], -max_translation_x, max_translation_x)
        ty = tf.random_uniform([num_batch], -max_translation_y, max_translation_y)
        rot = tf.random_uniform([num_batch], -max_rotation, max_rotation)
        rad = _deg2rad(rot)
        scale = tf.random_uniform([num_batch], min_scale, max_scale)

        t1 = [[tf.cos(rad), -tf.sin(rad), tx],
              [tf.sin(rad), tf.cos(rad), ty]] # H * W * B
        t1 = tf.transpose(t1, [2, 0, 1]) # B * H * W

        scale_x = scale
        if horizontal_flipping:
            flip = tf.random_uniform([num_batch], 0, 1)
            flip = tf.where(tf.greater(flip, 0.5), -one, one)
            scale_x = scale_x * flip

        t2 = [[scale_x, zero, zero],
              [zero, scale, zero],
              [zero, zero, one]] # H * W * B
        t2 = tf.transpose(t2, [2, 0, 1]) # B * H * W

        t = tf.matmul(t1, t2)

        out = []
        for tensor in tensors:
            shape = tf.shape(tensor)
            tensor = transformer(tensor, t, (shape[1], shape[2]))
            out.append(tf.stop_gradient(tensor))
    if intrinsics is not None:
        fx = intrinsics[:, 0, 0] * scale_x
        fy = intrinsics[:, 1, 1] * scale
        cx = intrinsics[:, 0, 2] * scale_x
        cy = intrinsics[:, 1, 2] * scale
        intrinsics = make_intrinsics_matrix(fx, fy, cx, cy)
        out.append(tf.stop_gradient(intrinsics))
    return out


def data_augmentation(im, intrinsics, out_h, out_w):
    # Random scaling
    def random_scaling(im, intrinsics):
        #batch_size, in_h, in_w, _ = im.get_shape().as_list()
        batch_size, in_h, in_w, _ = tf.unstack(tf.shape(im))
        scaling = tf.random_uniform([2], 1.0, 1.15)
        x_scaling = scaling[0]
        y_scaling = scaling[1]

        out_h = tf.cast(in_h * y_scaling, dtype=tf.int32)
        out_w = tf.cast(in_w * x_scaling, dtype=tf.int32)
        im = tf.image.resize_area(im, [out_h, out_w])
        fx = intrinsics[:, 0, 0] * x_scaling
        fy = intrinsics[:, 1, 1] * y_scaling
        cx = intrinsics[:, 0, 2] * x_scaling
        cy = intrinsics[:, 1, 2] * y_scaling
        intrinsics = make_intrinsics_matrix(fx, fy, cx, cy)
        return im, intrinsics

    # Random cropping
    def random_cropping(im, intrinsics, out_h, out_w):
        # batch_size, in_h, in_w, _ = im.get_shape().as_list()
        batch_size, in_h, in_w, _ = tf.unstack(tf.shape(im))
        offset_y = tf.random_uniform([1], 0, in_h - out_h + 1, dtype=tf.int32)[0]
        offset_x = tf.random_uniform([1], 0, in_w - out_w + 1, dtype=tf.int32)[0]
        im = tf.image.crop_to_bounding_box(
            im, offset_y, offset_x, out_h, out_w)
        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2] - tf.cast(offset_x, dtype=tf.float32)
        cy = intrinsics[:, 1, 2] - tf.cast(offset_y, dtype=tf.float32)
        intrinsics = make_intrinsics_matrix(fx, fy, cx, cy)
        return im, intrinsics

    im, intrinsics = random_scaling(im, intrinsics)
    im, intrinsics = random_cropping(im, intrinsics, out_h, out_w)
    return im, intrinsics


def random_photometric(ims, *,
                       noise_stddev=0.0, min_contrast=0.0, max_contrast=0.0,
                       brightness_stddev=0.0, min_colour=1.0, max_colour=1.0,
                       min_gamma=1.0, max_gamma=1.0):
    """Applies photometric augmentations to a list of image batches.

    Each image in the list is augmented in the same way.
    For all elements, num_batch must be equal while height and width may differ.

    Args:
        ims: list of 3-channel image batches normalized to [0, 1].
        channel_mean: tensor of shape [3] which was used to normalize the pixel
            values ranging from 0 ... 255.

    Returns:
        Batch of normalized images with photometric augmentations. Has the same
        shape as the input batch.
    """

    with tf.variable_scope('random_photometric'):
        num_batch = tf.shape(ims[0])[0]

        contrast = tf.random_uniform([num_batch, 1], min_contrast, max_contrast)
        gamma = tf.random_uniform([num_batch, 1], min_gamma, max_gamma)
        gamma_inv = 1.0 / gamma
        colour = tf.random_uniform([num_batch, 3], min_colour, max_colour)
        if noise_stddev > 0.0:
            noise = tf.random_normal([num_batch, 1], stddev=noise_stddev)
        else:
            noise = tf.zeros([num_batch, 1])
        if brightness_stddev > 0.0:
            brightness = tf.random_normal([num_batch, 1],
                                          stddev=brightness_stddev)
        else:
            brightness = tf.zeros([num_batch, 1])

        out = []
        for im in ims:
            # Transpose to [height, width, num_batch, channels]
            im_re = tf.transpose(im, [1, 2, 0, 3])
            im_re = im_re
            im_re = (im_re * (contrast + 1.0) + brightness) * colour
            im_re = tf.maximum(0.0, tf.minimum(1.0, im_re))
            im_re = tf.pow(im_re, gamma_inv)

            im_re = im_re + noise

            # Subtract the mean again after clamping
            im_re = im_re

            im = tf.transpose(im_re, [2, 0, 1, 3])
            im = tf.stop_gradient(im)
            out.append(im)
        return out

def random_crop(tensors, size, intrinsic=None, seed=None, name=None):
    """Randomly crops multiple tensors (of the same shape) to a given size.

    Each tensor is cropped in the same way."""
    with tf.name_scope(name, "random_crop", [size]) as name:
        size = tf.convert_to_tensor(size, dtype=tf.int32, name="size")
        if len(tensors) == 2:
            shape = tf.minimum(tf.shape(tensors[0]), tf.shape(tensors[1]))
        else:
            shape = tf.shape(tensors[0])

        limit = shape - size + 1
        offset = tf.random_uniform(
           tf.shape(shape),
           dtype=size.dtype,
           maxval=size.dtype.max,
           seed=seed) % limit
        results = []
        for tensor in tensors:
            result = tf.slice(tensor, offset, size)
            results.append(result)
        if intrinsic is not None:
            intrinsic = tf.cast(intrinsic, dtype=tf.float32)
            fx = intrinsic[0, 0]
            fy = intrinsic[1, 1]
            cx = intrinsic[0, 2] - tf.cast(offset[0], dtype=tf.float32)
            cy = intrinsic[1, 2] - tf.cast(offset[1], dtype=tf.float32)
            intrinsic = make_intrinsic_matrix(fx, fy, cx, cy)
            results.append(intrinsic)
        return results

def make_intrinsic_matrix(fx, fy, cx, cy):
    # Assumes batch input
    zero = tf.zeros_like(fx)
    r1 = tf.stack([fx, zero, cx], axis=0)
    r2 = tf.stack([zero, fy, cy], axis=0)
    r3 = tf.constant([0.,0.,1.], shape=[3])
    intrinsic = tf.stack([r1, r2, r3], axis=0)
    return intrinsic

def make_intrinsics_matrix(fx, fy, cx, cy):
    # Assumes batch input
    batch_size = fx.get_shape().as_list()[0]
    zeros = tf.zeros_like(fx)
    r1 = tf.stack([fx, zeros, cx], axis=1)
    r2 = tf.stack([zeros, fy, cy], axis=1)
    r3 = tf.constant([0.,0.,1.], shape=[1, 3])
    r3 = tf.tile(r3, [batch_size, 1])
    intrinsics = tf.stack([r1, r2, r3], axis=1)
    return intrinsics

def get_multi_scale_intrinsics(intrinsics, num_scales):
    intrinsics_mscale = []
    # Scale the intrinsics accordingly for each scale
    for s in range(num_scales):
        fx = intrinsics[:,0,0]/(2 ** s)
        fy = intrinsics[:,1,1]/(2 ** s)
        cx = intrinsics[:,0,2]/(2 ** s)
        cy = intrinsics[:,1,2]/(2 ** s)
        intrinsics_mscale.append(
            make_intrinsics_matrix(fx, fy, cx, cy))
    intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)
    return intrinsics_mscale

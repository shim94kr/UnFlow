import tensorflow as tf
import numpy as np


def summarized_placeholder(name, prefix=None, key=tf.GraphKeys.SUMMARIES):
    prefix = '' if not prefix else prefix + '/'
    p = tf.placeholder(tf.float32, name=name)
    tf.summary.scalar(prefix + name, p, collections=[key])
    return p


def resize_area(tensor, like):
    _, h, w, _ = like.get_shape().as_list()
    return tf.stop_gradient(tf.image.resize_area(tensor, [h, w]))


def resize_bilinear(tensor, like):
    _, h, w, _ = like.get_shape().as_list()
    return tf.stop_gradient(tf.image.resize_bilinear(tensor, [h, w]))


def make_grid(batch_size, H, W):
    rowx1 = tf.range(W)  # W
    rowxH = tf.cast(tf.reshape(tf.tile(rowx1, [H]), [1, H, W]), tf.float32)  # 1 * H * W
    colx1 = tf.expand_dims(tf.range(H), 1)  # H
    colxW = tf.cast(tf.reshape(tf.tile(colx1, [1, W]), [1, H, W]), tf.float32)  # 1 * H * W
    ones_cnst = tf.ones(shape=[1, H, W])
    grid = tf.tile(tf.stack([rowxH, colxW, ones_cnst], axis=3), [batch_size, 1, 1, 1]) # B * H * W * 3

    return grid

def euler2mat(z, y, x):
  """Converts euler angles to rotation matrix
   TODO: remove the dimension for 'N' (deprecated for converting all source
         poses altogether)
   Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
  Args:
      z: rotation angle along z axis (in radians) -- size = [B, N]
      y: rotation angle along y axis (in radians) -- size = [B, N]
      x: rotation angle along x axis (in radians) -- size = [B, N]
  Returns:
      Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
  """
  B = tf.shape(z)[0]
  N = 1
  z = tf.clip_by_value(z, -np.pi, np.pi)
  y = tf.clip_by_value(y, -np.pi, np.pi)
  x = tf.clip_by_value(x, -np.pi, np.pi)

  # Expand to B x N x 1 x 1
  z = tf.expand_dims(tf.expand_dims(z, -1), -1)
  y = tf.expand_dims(tf.expand_dims(y, -1), -1)
  x = tf.expand_dims(tf.expand_dims(x, -1), -1)

  zeros = tf.zeros([B, N, 1, 1])
  ones  = tf.ones([B, N, 1, 1])

  cosz = tf.cos(z)
  sinz = tf.sin(z)
  rotz_1 = tf.concat([cosz, -sinz, zeros], axis=3)
  rotz_2 = tf.concat([sinz,  cosz, zeros], axis=3)
  rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
  zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)

  cosy = tf.cos(y)
  siny = tf.sin(y)
  roty_1 = tf.concat([cosy, zeros, siny], axis=3)
  roty_2 = tf.concat([zeros, ones, zeros], axis=3)
  roty_3 = tf.concat([-siny,zeros, cosy], axis=3)
  ymat = tf.concat([roty_1, roty_2, roty_3], axis=2)

  cosx = tf.cos(x)
  sinx = tf.sin(x)
  rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
  rotx_2 = tf.concat([zeros, cosx, -sinx], axis=3)
  rotx_3 = tf.concat([zeros, sinx, cosx], axis=3)
  xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)

  rotMat = tf.matmul(tf.matmul(xmat, ymat), zmat)
  return rotMat

def eulergrid2mat(z, y, x):
  """Converts euler angles to rotation matrix
   TODO: remove the dimension for 'N' (deprecated for converting all source
         poses altogether)
   Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
  Args:
      z: rotation angle along z axis (in radians) -- size = [B, H, W]
      y: rotation angle along y axis (in radians) -- size = [B, H, W]
      x: rotation angle along x axis (in radians) -- size = [B, H, W]
  Returns:
      Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
  """

  B, H, W, _ = tf.unstack(tf.shape(z))
  z = tf.clip_by_value(z, -np.pi, np.pi)
  y = tf.clip_by_value(y, -np.pi, np.pi)
  x = tf.clip_by_value(x, -np.pi, np.pi)

  # Expand to B x H x W x 1 x 1
  z = tf.expand_dims(z, -1)
  y = tf.expand_dims(y, -1)
  x = tf.expand_dims(x, -1)

  zeros = tf.zeros([B, H, W, 1, 1])
  ones  = tf.ones([B, H, W, 1, 1])

  cosz = tf.cos(z)
  sinz = tf.sin(z)
  rotz_1 = tf.concat([cosz, -sinz, zeros], axis=4)
  rotz_2 = tf.concat([sinz,  cosz, zeros], axis=4)
  rotz_3 = tf.concat([zeros, zeros, ones], axis=4)
  zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=3)

  cosy = tf.cos(y)
  siny = tf.sin(y)
  roty_1 = tf.concat([cosy, zeros, siny], axis=4)
  roty_2 = tf.concat([zeros, ones, zeros], axis=4)
  roty_3 = tf.concat([-siny,zeros, cosy], axis=4)
  ymat = tf.concat([roty_1, roty_2, roty_3], axis=3)

  cosx = tf.cos(x)
  sinx = tf.sin(x)
  rotx_1 = tf.concat([ones, zeros, zeros], axis=4)
  rotx_2 = tf.concat([zeros, cosx, -sinx], axis=4)
  rotx_3 = tf.concat([zeros, sinx, cosx], axis=4)
  xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=3)

  rotMat = tf.matmul(tf.matmul(xmat, ymat), zmat) # B * H * W * 3 * 3
  return rotMat

def posegrid_vec2mat(posegrid):
  """Converts 6DoF parameters to transformation matrix
  Args:
      vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, H, W, 6]
  Returns:
      rotation grid -- [B, H, W, 3, 3], translation grid -- [B, H, W, 3]
  """
  batch_size, H, W, _ = tf.unstack(tf.shape(posegrid))
  sigtx = tf.exp(-tf.nn.relu(tf.slice(posegrid, [0, 0, 0, 6], [-1, -1, -1, 1])))
  sigty = tf.exp(-tf.nn.relu(tf.slice(posegrid, [0, 0, 0, 7], [-1, -1, -1, 1])))
  sigtz = tf.exp(-tf.nn.relu(tf.slice(posegrid, [0, 0, 0, 8], [-1, -1, -1, 1])))
  sigr = tf.exp(-tf.nn.relu(tf.slice(posegrid, [0, 0, 0, 9], [-1, -1, -1, 1])))
  tx = tf.slice(posegrid, [0, 0, 0, 0], [-1, -1, -1, 1]) * sigtx
  ty = tf.slice(posegrid, [0, 0, 0, 1], [-1, -1, -1, 1]) * sigty
  tz = tf.slice(posegrid, [0, 0, 0, 2], [-1, -1, -1, 1]) * sigtz
  rx = tf.slice(posegrid, [0, 0, 0, 3], [-1, -1, -1, 1]) * sigr
  ry = tf.slice(posegrid, [0, 0, 0, 4], [-1, -1, -1, 1]) * sigr
  rz = tf.slice(posegrid, [0, 0, 0, 5], [-1, -1, -1, 1]) * sigr
  translation = tf.concat([tx, ty, tz], axis = 3)
  rot_mat = eulergrid2mat(rz, ry, rx)
  return rot_mat, translation, tf.concat([sigtx, sigty, sigtz, sigr], axis=3), tf.concat([translation, rx, ry, rz], axis=3)

def pose_vec2mat(vec):
  """Converts 6DoF parameters to transformation matrix
  Args:
      vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
  Returns:
      A transformation matrix -- [B, 4, 4]
  """
  batch_size, _ = tf.unstack(tf.shape(vec))
  translation = tf.slice(vec, [0, 0], [-1, 3])
  translation = tf.expand_dims(translation, -1)
  rx = tf.slice(vec, [0, 3], [-1, 1])
  ry = tf.slice(vec, [0, 4], [-1, 1])
  rz = tf.slice(vec, [0, 5], [-1, 1])
  rot_mat = euler2mat(rz, ry, rx)
  rot_mat = tf.squeeze(rot_mat, axis=[1])
  filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
  filler = tf.tile(filler, [batch_size, 1, 1])
  transform_mat = tf.concat([rot_mat, translation], axis=2)
  transform_mat = tf.concat([transform_mat, filler], axis=1)
  return transform_mat

import tensorflow as tf
import tensorflow.contrib.slim as slim

# Definition of the discriminator
def discriminator(dis_inputs, is_training=None, reuse = False):

    # Define the discriminator block
    def discriminator_block(inputs, output_channel, kernel_size, stride, scope):
        with tf.variable_scope(scope):
            net = conv2(inputs, kernel_size, output_channel, stride, use_bias=False, scope='conv1')
            net = batchnorm(net, is_training)
            net = lrelu(net, 0.2)

        return net

    with tf.variable_scope('discriminator_unit', reuse=reuse):
        # The input layer
        with tf.variable_scope('input_stage'):
            net = conv2(dis_inputs, 3, 64, 1, scope='conv')
            net = lrelu(net, 0.2)

        # The discriminator block part
        # block 1
        net = discriminator_block(net, 64, 3, 2, 'disblock_1')

        # block 2
        net = discriminator_block(net, 128, 3, 1, 'disblock_2')

        # block 3
        net = discriminator_block(net, 128, 3, 2, 'disblock_3')

        # block 4
        net = discriminator_block(net, 256, 3, 1, 'disblock_4')

        # block 5
        net = discriminator_block(net, 256, 3, 2, 'disblock_5')

        # block 6
        net = discriminator_block(net, 512, 3, 1, 'disblock_6')

        # block_7
        net = discriminator_block(net, 512, 3, 2, 'disblock_7')

        # The dense layer 1
        with tf.variable_scope('dense_layer_1'):
            net = slim.flatten(net)
            net = denselayer(net, 1024)
            net = lrelu(net, 0.2)

        # The dense layer 2
        with tf.variable_scope('dense_layer_2'):
            net = denselayer(net, 1)
            net = tf.nn.sigmoid(net)

    return net

# Define the convolution building block
def conv2(batch_input, kernel=3, output_channel=64, stride=1, use_bias=True, scope='conv'):
    # kernel: An integer specifying the width and height of the 2D convolution window
    with tf.variable_scope(scope):
        if use_bias:
            return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
                            activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
        else:
            return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
                            activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=None)

def batchnorm(inputs, is_training):
    return slim.batch_norm(inputs, decay=0.9, epsilon=0.001, updates_collections=tf.GraphKeys.UPDATE_OPS,
                        scale=False, fused=True, is_training=is_training)

def lrelu(inputs, alpha):
    with tf.variable_scope('leaky_relu'):
        return tf.maximum(alpha * inputs, inputs)

# Our dense layer
def denselayer(inputs, output_size):
    output = tf.layers.dense(inputs, output_size, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
    return output
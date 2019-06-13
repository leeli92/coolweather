import tensorflow as tf
from tensorflow.contrib import slim

def ResnetBlock(x, dim, ksize, scope='rb'):
    with tf.variable_scope(scope):
        net = slim.conv2d(x, dim, [ksize, ksize],activation_fn=tf.nn.leaky_relu, scope='conv1')
        net = slim.conv2d(net, dim, [ksize, ksize], activation_fn=None, scope='conv2')
        return net + x


def bi_block(x, outchannel, name):
    with tf.variable_scope(name):
        net = slim.conv2d(x, outchannel, [5, 5], stride=2, activation_fn=tf.nn.leaky_relu)
        net_1 = slim.max_pool2d(x, [5, 5], padding='SAME')
    return  net + net_1
def upsampling2d(x, size=(2, 2), name= 'upsampling'):
    with tf.name_scope(name):
        shape = x.get_shape().as_list()
        return tf.image.resize_bilinear(x, size=(size[0] * shape[1], size[1] * shape[2]))
def generator(x, reuse=False, scope='generator'):
    # encode model stage_1
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                activation_fn=tf.nn.leaky_relu, padding='SAME', normalizer_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                biases_initializer=tf.constant_initializer(0.0)):
            shape = x.get_shape().as_list()
            conv1_1 = slim.conv2d(x, 8, kernel_size=[3, 3], stride=2)
            # bi_block
            conv2_2 = ResnetBlock(conv1_1, 8, 3,scope="conv2_2")
            # bi_block
            conv3_0 = slim.conv2d(conv2_2, 16, kernel_size=[3, 3],stride=2)
            conv3_2 = ResnetBlock(conv3_0, 16, 3,scope="conv3_2")

            filter1 = tf.get_variable('filter1', [5, 5, 16, 16], initializer=tf.random_normal_initializer(stddev=0.02))
            deconv1_2 = tf.nn.conv2d_transpose(conv3_2, filter1,
                output_shape=[shape[0], shape[1] //2, shape[2] // 2, 16], strides=[1, 2, 2, 1], name="deconv1_2")
            filter2 = tf.get_variable('filter2', [5, 5, 8, 16], initializer=tf.random_normal_initializer(stddev=0.02))
            deconv2_2 = tf.nn.conv2d_transpose(deconv1_2, filter2,
                output_shape=[shape[0], shape[1], shape[2], 8], strides=[1, 2, 2, 1], name="deconv2_2")
            output = slim.conv2d(deconv2_2, 3, kernel_size=[3, 3], activation_fn=tf.nn.sigmoid,scope="output")
            return output
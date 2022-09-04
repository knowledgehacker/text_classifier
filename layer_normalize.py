import config

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


class LayerNormalization(object):
    def __init__(self):
        self.embed_dim = config.EMBED_SIZE

        with tf.variable_scope("layer_norm"):
            self.epsilon = 1e-6
            self.scale = tf.get_variable(
                "scale",
                [self.embed_dim], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.bias = tf.get_variable(
                "bias",
                [self.embed_dim], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.01))

    def normalize(self, input):
        mean = tf.reduce_mean(input, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(input - mean), axis=[-1], keepdims=True)
        normalized_input = (input - mean) / tf.square(variance + self.epsilon)
        normalized_input = normalized_input * self.scale + self.bias

        return normalized_input

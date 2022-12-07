import config

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


class TextCNN(object):
    def __init__(self):
        with tf.variable_scope("embed"):
            self.word_embed_matrix = tf.get_variable('word_embed_matrix', [config.VOCAB_SIZE, config.EMBED_SIZE], dtype=tf.float32,
                                                     initializer=tf.truncated_normal_initializer(stddev=0.01))

    def forward(self, input, dropout_keep_prob_ph):
        with tf.device('/cpu:0'):
            embed = tf.nn.embedding_lookup(self.word_embed_matrix, input, name='embedding_lookup')

        pooled_outputs = []
        for conv_filter_kernel_size in config.CONV_FILTER_KERNEL_SIZES:
            with tf.variable_scope('cnn-%d' % conv_filter_kernel_size):
                # conv layer, relu activation ensures elements of output > 0
                conv = tf.layers.conv1d(embed,
                                        config.CONV_FILTER_NUM,
                                        conv_filter_kernel_size,
                                        activation=tf.nn.relu,
                                        name='conv-%d' % conv_filter_kernel_size)
                # global max pooling layer, max over a convolution filter
                gmp = tf.reduce_max(conv, axis=1, name='gmp-%d' % conv_filter_kernel_size)
                pooled_outputs.append(gmp)

        # dimension of h_pool: [config.BATCH_SIZE, CONV_FILTER_NUM * CONV_FILTER_KERNEL_SIZES]
        h_pool = tf.concat(pooled_outputs, 1)
        h_dropout = tf.nn.dropout(h_pool, rate=1.0 - dropout_keep_prob_ph)

        return h_dropout

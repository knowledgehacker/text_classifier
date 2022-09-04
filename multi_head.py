# -*- coding: utf-8 -*-
import config

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


class AttentionHead(object):
    def __init__(self, head_dim, layer_index, head_index):
        self.embed_dim = config.EMBED_SIZE

        with tf.variable_scope("multi-head_%s__%s" % (layer_index, head_index)):
            self.q = tf.get_variable(name="query", shape=[self.embed_dim, head_dim], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.k = tf.get_variable(name="key", shape=[self.embed_dim, head_dim], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.v = tf.get_variable(name="value", shape=[self.embed_dim, head_dim], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.01))

    # input_embed is of shape [batch_size, seq_len, embed_size], seq_len is the length of sequence
    def forward(self, input_embed):
        # the dimension of query, key, value is head_dim = embed_size / num_attention_heads
        query = tf.matmul(input_embed, self.q)
        key = tf.matmul(input_embed, self.k)
        value = tf.matmul(input_embed, self.v)

        n = tf.shape(key)[-1]   # = tf.shape(input_embed)[1]

        # score is of shape [seq_len, seq_len], ignoring batch_size
        scores = tf.matmul(query, key, transpose_b=True) / tf.sqrt(tf.to_float(n))
        weights = tf.nn.softmax(scores)
        # attention_weighted_embed is of shape[seq_len, embed_size], ignoring batch_size, is value weighted by weights
        attention_weighted_embed = tf.matmul(weights, value)

        return attention_weighted_embed


class MultiHeadAttention(object):
    def __init__(self, layer_index):
        self.embed_dim = config.EMBED_SIZE

        head_dim = int(self.embed_dim / config.NUM_ATTENTION_HEAD)
        self.heads = [AttentionHead(head_dim, layer_index, head_index) for head_index in range(config.NUM_ATTENTION_HEAD)]

    def forward(self, input_embed):
        # each head is of shape [seq_len, head_dim], ignoring batch_size, after concat, head_dim changed to embed_size
        multi_heads = tf.concat([head.forward(input_embed) for head in self.heads], axis=-1)
        linearized_multi_heads = tf.layers.dense(multi_heads, self.embed_dim)

        return linearized_multi_heads

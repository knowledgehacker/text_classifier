# -*- coding: utf-8 -*-

import config
from multi_head import MultiHeadAttention

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


class PositionFFN(object):
    def __init__(self, layer_index):
        self.layer_index = layer_index

        self.embed_dim = config.EMBED_SIZE

        with tf.variable_scope("encoder-layer_%s" % layer_index):
            self.w_1 = tf.get_variable(name="w_1", shape=[self.embed_dim, self.embed_dim * 4], dtype=tf.float32,
                                       initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.b_1 = tf.get_variable(name="b_1", shape=[self.embed_dim * 4], dtype=tf.float32,
                                       initializer=tf.truncated_normal_initializer(stddev=0.01))

            self.w_2 = tf.get_variable(name="w_2", shape=[self.embed_dim * 4, self.embed_dim], dtype=tf.float32,
                                       initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.b_2 = tf.get_variable(name="b_2", shape=[self.embed_dim], dtype=tf.float32,
                                       initializer=tf.truncated_normal_initializer(stddev=0.01))

    def forward(self, multi_heads_output, dropout_keep_prob_ph):
        h1 = tf.add(tf.matmul(multi_heads_output, self.w_1), self.b_1)
        h1 = tf.nn.relu(h1)

        h2 = tf.add(tf.matmul(h1, self.w_2), self.b_2)
        h2 = tf.nn.dropout(h2, rate=1 - dropout_keep_prob_ph)

        return h2


# TODO: 我们怎么将输入文本分解成一个个句子?
class EncoderLayer(object):
    def __init__(self, layer_index):
        self.layer_index = layer_index

        self.vocab_size = config.VOCAB_SIZE
        self.embed_dim = config.EMBED_SIZE

        with tf.variable_scope("encoder-layer_%s" % layer_index):
            self.layer_normalize_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.layer_normalize_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.multi_heads_attention = MultiHeadAttention(layer_index)
        self.position_ffn = PositionFFN(layer_index)

    """
    # each layer has its own scale and bias parameters when normalization
    def layer_normalize(self, embed, layer_name):
        with tf.variable_scope("layer_norm/%s" % layer_name):
            epsilon = 1e-6
            scale = tf.get_variable(
                "scale",
                [self.embed_dim], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.01))
            bias = tf.get_variable(
                "bias",
                [self.embed_dim], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.01))

        mean = tf.reduce_mean(embed, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(embed - mean), axis=[-1], keepdims=True)
        normalized_embed = (embed - mean) / tf.square(variance + epsilon)
        normalized_embed = normalized_embed * scale + bias

        return normalized_embed
    """

    def forward(self, embed_dropout, dropout_keep_prob_ph):
        # multi-heads attention network
        #normalized_embed = self.layer_normalize(embed_dropout, "embed_dropout")
        normalized_embed = self.layer_normalize_1(embed_dropout)
        multi_heads_output = embed_dropout + self.multi_heads_attention.forward(normalized_embed)

        print("--- %s multi_heads_output" % self.layer_index)
        print(multi_heads_output)

        # position-wise feed forward network for encoder
        #normalized_multi_heads_output = self.layer_normalize(multi_heads_output, "multi_heads_output")
        normalized_multi_heads_output = self.layer_normalize_2(multi_heads_output)
        position_ffn_output = multi_heads_output + self.position_ffn.forward(normalized_multi_heads_output, dropout_keep_prob_ph)

        print("--- %s position_ffn_output" % self.layer_index)
        print(position_ffn_output)

        return position_ffn_output


class Encoder(object):
    def __init__(self):
        self.num_encoder_layer = config.NUM_ENCODER_LAYER

        self.vocab_size = config.VOCAB_SIZE
        self.embed_dim = config.EMBED_SIZE

        with tf.variable_scope("embed"):
            self.word_embed_matrix = tf.get_variable(
                "word_embed_matrix",
                [self.vocab_size, self.embed_dim], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.01))

            self.position_embed_matrix = tf.get_variable(
                "position_embed_matrix",
                [self.vocab_size, self.embed_dim], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.01))

        self.layer_normalize_0 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.encoder_layers = [EncoderLayer(layer_index) for layer_index in range(self.num_encoder_layer)]

    def forward(self, input_encode, dropout_keep_prob_ph):
        # embed input word and position, we do not have mask encode as input here

        print("--- input_encode")
        print(input_encode)

        """
        # TODO: something wrong with position encoding?
        seq_length = tf.shape(input_encode)[1]
        position_encode = tf.expand_dims(tf.range(seq_length, dtype=tf.int64), axis=0)

        print("--- position_encode")
        print(position_encode)

        word_embed = tf.nn.embedding_lookup(self.word_embed_matrix, input_encode, name='word_embed_lookup')
        position_embed = tf.nn.embedding_lookup(self.position_embed_matrix, position_encode, name='position_embed_lookup')
        embed = word_embed + position_embed
        """

        word_embed = tf.nn.embedding_lookup(self.word_embed_matrix, input_encode, name='word_embed_lookup')
        embed = word_embed

        print("--- embed")
        print(embed)

        #embed = self.layer_normalize(embed, "embed")
        embed = self.layer_normalize_0(embed)
        embed_dropout = tf.nn.dropout(embed, rate=1 - dropout_keep_prob_ph)

        # transformer encoder stack
        encoder_output = embed_dropout
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer.forward(encoder_output, dropout_keep_prob_ph)

        return encoder_output


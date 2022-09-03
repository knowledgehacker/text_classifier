# -*- coding: utf-8 -*-

import config
from multi_head import MultiHeadAttention

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


class PositionFFN(object):

    def forward(self, multi_heads_output, dropout_keep_prob_ph):
        with tf.variable_scope('position'):
            h1 = tf.layers.dense(multi_heads_output, config.EMBED_SIZE * 4, name='h1')
            h1 = tf.nn.relu(h1)

            h2 = tf.layers.dense(h1, config.EMBED_SIZE, name='h2')
            h2 = tf.nn.dropout(h2, dropout_keep_prob_ph)

        return h2


# TODO: 我们怎么将输入文本分解成一个个句子?
class Encoder(object):
    def __init__(self):
        self.num_class = config.NUM_CLASS

        self.vocab_size = config.VOCAB_SIZE
        self.embed_dim = config.EMBED_SIZE

        with tf.variable_scope("embed"):
            self.word_embed_matrix = tf.get_variable(
                "word_embed_matrix",
                [self.vocab_size, self.embed_dim], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.01))

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

        embed = self.layer_normalize(embed, "embed")
        embed_dropout = tf.nn.dropout(embed, rate=dropout_keep_prob_ph)

        # transformer encoder stack, here the stack consists of only one single transformer encoder
        normalized_embed = self.layer_normalize(embed_dropout, "embed_dropout")
        multi_heads_attention = MultiHeadAttention()
        multi_heads_output = embed_dropout + multi_heads_attention.forward(normalized_embed)

        print("--- multi_heads_output")
        print(multi_heads_output)

        # position-wise feed forward network for encoder
        normalized_multi_heads_output = self.layer_normalize(multi_heads_output, "multi_heads_output")
        position_ffn = PositionFFN()
        position_ffn_output = multi_heads_output + position_ffn.forward(normalized_multi_heads_output, dropout_keep_prob_ph)

        print("--- position_ffn_output")
        print(position_ffn_output)

        # classifier, transformer encoder output of the first word ([CLS]) for each sentence is used for classification.
        first_word = position_ffn_output[:, 0, :]
        print("--- first_word")
        print(first_word)

        first_word = tf.nn.dropout(first_word, rate=dropout_keep_prob_ph)
        with tf.variable_scope('classifier'):
            output = tf.layers.dense(first_word, config.NUM_CLASS, name='output')

        print("--- output")
        print(output)

        return output

    def predict(self, logits, label):
        with tf.name_scope("predict"):
            # prediction
            #preds = tf.argmax(tf.nn.softmax(logits), 1, name='predictions')
            preds = tf.argmax(logits, 1, name='predictions')
            # accuracy
            correct_preds = tf.equal(tf.argmax(label, 1), preds)
            acc = tf.reduce_mean(tf.cast(correct_preds, tf.float32), name='accuracy')

        return preds, acc

    def opt(self, logits, label):
        with tf.name_scope("loss"):
            # loss
            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))
            train_op = tf.train.AdamOptimizer(learning_rate=config.LEARNING_RATE).minimize(loss_op)

        return loss_op, train_op

import config
from encoder import Encoder

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


class Classifier(object):
    def __init__(self):
        self.encoder = Encoder()

        with tf.variable_scope("classifier"):
            self.w = tf.get_variable(name="w", shape=[config.EMBED_SIZE, config.NUM_CLASS], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.b = tf.get_variable(name="b", shape=[config.NUM_CLASS], dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.01))

    def forward(self, input_encode, dropout_keep_prob_ph):
        # classifier, transformer encoder output of the first word ([CLS]) for each sentence is used for classification.
        encoder_output = self.encoder.forward(input_encode, dropout_keep_prob_ph)

        first_word = encoder_output[:, 0, :]
        print("--- first_word")
        print(first_word)

        first_word = tf.nn.dropout(first_word, rate=1 - dropout_keep_prob_ph)
        """
        with tf.variable_scope('classifier'):
            output = tf.layers.dense(first_word, config.NUM_CLASS, name='output')
        """
        output = tf.add(tf.matmul(first_word, self.w), self.b)

        print("--- output")
        print(output)

        return output

    def predict(self, logits, label):
        with tf.name_scope("output"):
            # prediction
            # preds = tf.argmax(tf.nn.softmax(logits), 1, name='predictions')
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

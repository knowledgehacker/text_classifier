# -*- coding: utf-8 -*-

import config
from input_feed import create_dataset
from classifier import EncoderClassifier, TextCNNClassifier
from utils import current_time, restore_sentence_size, save_model, get_optimizer

"""
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
"""

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
cfg.gpu_options.allow_growth = True

"""
if we create model in main() and pass it to train(), instead of in train(), we will get error as follows.
ValueError: Tensor("embedding/word_embed_matrix:0", shape=(50000, 64), dtype=float32_ref) must be from the same graph as 
Tensor("content_ph:0", shape=(?, 1000), dtype=int64) (graphs are <tensorflow.python.framework.ops.Graph object at 0x10ff34b50> 
and <tensorflow.python.framework.ops.Graph object at 0x10ff346d0>).
"""


def train():
    print(current_time(), "training starts...")

    # sentence_size_max = restore_sentence_size(config.SENTENCE_FILE)
    sentence_size_max = config.SENTENCE_SIZE_MAX
    print("sentence_size_max=%d" % sentence_size_max)

    g = tf.Graph()
    with g.as_default():
        # create a feed-able iterator, to be feed by train and test datasets
        handle_ph = tf.placeholder(dtype=tf.string, name="handle_ph")
        train_dataset = create_dataset(config.TF_TRAIN_PATH, sentence_size_max, test=False)
        train_iterator = tf.data.make_initializable_iterator(train_dataset)
        #train_iterator = train_dataset.make_initializable_iterator()

        """
        test_dataset = create_dataset(config.TF_TEST_PATH, sentence_size_max, test=True)
        test_iterator = tf.data.make_initializable_iterator(test_dataset)
        test_iterator = test_dataset.make_initializable_iterator()
        """

        iterator = tf.data.Iterator.from_string_handle(
            handle_ph,
            tf.data.get_output_types(train_dataset),
            tf.data.get_output_shapes(train_dataset),
            tf.data.get_output_classes(train_dataset))
        content, label = iterator.get_next(name="next_batch")

        # create model network
        #To be able to feed with batches of different size, the first dimension should be None
        content_ph = tf.placeholder(dtype=tf.int64, shape=(None, sentence_size_max), name="content_ph")
        label_ph = tf.placeholder(dtype=tf.float32, shape=(None, config.NUM_CLASS), name="label_ph")

        dropout_keep_prob_ph = tf.placeholder(tf.float32, name="dropout_keep_prob")

        if config.MODEL_NAME == "encoder":
            model = EncoderClassifier()
        elif config.MODEL_NAME == "textcnn":
            model = TextCNNClassifier()
        else:
            print("Unsupported model %s" % config.MODEL_NAME)
            exit(-1)

        logits = model.forward(content_ph, dropout_keep_prob_ph)
        loss_op = model.loss(logits, label_ph)
        opt = get_optimizer()
        train_op = opt.minimize(loss_op)
        preds_op, acc_op = model.predict(logits, label_ph)

        # create saver
        saver = tf.train.Saver(max_to_keep=1)

    with tf.Session(graph=g, config=cfg) as sess:
        tf.global_variables_initializer().run()

        # create handle to feed to iterator's string_handle placeholder
        train_handle = sess.run(train_iterator.string_handle())
        #test_handle = sess.run(test_iterator.string_handle())

        step = 0
        for i in range(config.NUM_EPOCH):
            print(current_time(), "epoch: %d" % (i + 1))
            sess.run(train_iterator.initializer)

            while True:
                try:
                    content_ts, label_ts = sess.run([content, label], feed_dict={handle_ph: train_handle})
                    _, train_acc, train_loss = sess.run([train_op, acc_op, loss_op],
                                                        feed_dict={content_ph: content_ts,
                                                                   label_ph: label_ts,
                                                                   dropout_keep_prob_ph: config.TRAIN_KEEP_PROB})

                    if step % config.STEPS_PER_CKPT == 0:
                        print(current_time(),
                              "step: %d, train_loss: %.3f, train_acc: %.3f" % (step, train_loss, train_acc))

                        saver.save(sess, config.CKPT_PATH, global_step=step)

                    step += 1
                except tf.errors.OutOfRangeError:
                    print(current_time(),
                          "step: %d, train_loss: %.3f, train_acc: %.3f" % (step, train_loss, train_acc))

                    saver.save(sess, config.CKPT_PATH, global_step=step)
                    break

            # save model
            outputs = ["logits", "loss", "output/predictions", "output/accuracy"]
            save_model(sess, config.MODLE_DIR, config.MODEL_NAME, outputs)

    print(current_time(), "training finishes...")


def main():
    # train
    train()


if __name__ == "__main__":
    main()

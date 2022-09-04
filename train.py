# -*- coding: utf-8 -*-

import config
from input_feed import create_dataset
from classifier import EncoderClassifier, TextCNNClassifier
from utils import current_time, cal_avg, restore_sentence_size

"""
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
"""

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

CKPT_PATH = '%s/%s' % (config.CKPT_DIR, config.MODEL_NAME)


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

        test_dataset = create_dataset(config.TF_TEST_PATH, sentence_size_max, test=True)
        test_iterator = tf.data.make_initializable_iterator(test_dataset)
        #test_iterator = test_dataset.make_initializable_iterator()

        iterator = tf.data.Iterator.from_string_handle(
            handle_ph,
            tf.data.get_output_types(train_dataset),
            tf.data.get_output_shapes(train_dataset),
            tf.data.get_output_classes(train_dataset))
        content, label = iterator.get_next("next_batch")

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
        loss_op, train_op = model.opt(logits, label_ph)
        preds_op, acc_op = model.predict(logits, label_ph)

        # create saver
        saver = tf.train.Saver()

    with tf.Session(graph=g, config=cfg) as sess:
        """
        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", loss_op)
        acc_summary = tf.summary.scalar("accuracy", acc_op)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_writer = tf.summary.FileWriter('logs/train_summaries.txt', sess.graph)
        """

        tf.global_variables_initializer().run()

        # create handle to feed to iterator's string_handle placeholder
        train_handle = sess.run(train_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())

        step = 0
        for i in range(config.NUM_EPOCH):
            print(current_time(), "epoch: %d" % (i + 1))
            sess.run(train_iterator.initializer)

            while True:
                try:
                    """
                    #train_acc, train_loss, train_summaries = train_batch(sess, train_op, acc_op, loss_op,
                    train_acc, train_loss = train_batch(sess, train_op, acc_op, loss_op,
                                                        handle_ph, content_ph, label_ph, dropout_keep_prob_ph,
                                                        train_handle, content, label)
                    """
                    content_ts, label_ts = sess.run([content, label], feed_dict={handle_ph: train_handle})

                    # _, acc, loss, train_summaries = sess.run([train_op, acc_op, loss_op, train_summary_op],
                    _, train_acc, train_loss = sess.run([train_op, acc_op, loss_op],
                                                        feed_dict={content_ph: content_ts,
                                                                   label_ph: label_ts,
                                                                   dropout_keep_prob_ph: config.TRAIN_KEEP_PROB})

                    # train_summary_writer.add_summary(train_summaries, step)

                    if step % config.STEPS_PER_CKPT == 0:
                        if config.VALIDATE:
                            test_avg_loss, test_avg_acc = validate(sess, test_iterator, preds_op, acc_op, loss_op,
                                     handle_ph, content_ph, label_ph, dropout_keep_prob_ph,
                                     test_handle, content, label)
                            #print("test_avg_loss: %.3f, test_avg_acc: %.3f" % (test_avg_loss, test_avg_acc))

                            print(current_time(),
                                  "step: %d, train_loss: %.3f, train_acc: %.3f | test_loss: %.3f, test_acc: %.3f" %
                                  (step, train_loss, train_acc, test_avg_loss, test_avg_acc))
                        else:
                            print(current_time(),
                                  "step: %d, train_loss: %.3f, train_acc: %.3f" % (step, train_loss, train_acc))

                        saver.save(sess, CKPT_PATH, global_step=step)

                    step += 1
                except tf.errors.OutOfRangeError:

                    if config.VALIDATE:
                        test_avg_loss, test_avg_acc = validate(sess, test_iterator, preds_op, acc_op, loss_op,
                                                               handle_ph, content_ph, label_ph, dropout_keep_prob_ph,
                                                               test_handle, content, label)
                        #print("test_avg_loss: %.3f, test_avg_acc: %.3f" % (test_avg_loss, test_avg_acc))

                        print(current_time(),
                              "step: %d, train_loss: %.3f, train_acc: %.3f | test_loss: %.3f, test_acc: %.3f" %
                              (step, train_loss, train_acc, test_avg_loss, test_avg_acc))
                    else:
                        print(current_time(),
                              "step: %d, train_loss: %.3f, train_acc: %.3f" % (step, train_loss, train_acc))

                    saver.save(sess, CKPT_PATH, global_step=step)
                    break

    print(current_time(), "training finishes...")


def train_batch(sess, train_op, acc_op, loss_op, # train_summery_op
                handle_ph, content_ph, label_ph, dropout_keep_prob_ph,
                train_handle, content, label):
    content_ts, label_ts = sess.run([content, label], feed_dict={handle_ph: train_handle})

    # _, acc, loss, train_summaries = sess.run([train_op, acc_op, loss_op, train_summary_op],
    _, train_acc, train_loss = sess.run([train_op, acc_op, loss_op],
                                        feed_dict={content_ph: content_ts,
                                                   label_ph: label_ts,
                                                   dropout_keep_prob_ph: config.TRAIN_KEEP_PROB})

    return train_acc, train_loss#, train_summaries


# batches get high accuracy from first to last, since the test cases are ordered by news categories
def validate(sess, test_iterator, preds_op, acc_op, loss_op,
             handle_ph, content_ph, label_ph, dropout_keep_prob_ph,
             test_handle, content, label):
    sess.run(test_iterator.initializer)

    test_losses = []
    test_accs = []

    test_step = 0
    while True:
        try:
            content_ts, label_ts = sess.run([content, label], feed_dict={handle_ph: test_handle})

            test_preds, test_acc, test_loss = sess.run([preds_op, acc_op, loss_op],
                                                       feed_dict={content_ph: content_ts,
                                                                  label_ph: label_ts,
                                                                  dropout_keep_prob_ph: config.TEST_KEEP_PROB})

            sample_num = len(test_preds)
            test_losses.append((test_loss, sample_num))
            test_accs.append((test_acc, sample_num))
            #print(current_time(), "test_step: %d, test_loss: %.3f, test_acc: %.3f" % (test_step, test_loss, test_acc))

            test_step += 1
        except tf.errors.OutOfRangeError:
            #print(current_time(), "test_step: %d, test_loss: %.3f, test_acc: %.3f" % (test_step, test_loss, test_acc))
            break

    test_avg_loss = cal_avg(test_losses)
    test_avg_acc = cal_avg(test_accs)

    return test_avg_loss, test_avg_acc


def main():
    # train
    train()


if __name__ == "__main__":
    main()

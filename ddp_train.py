# -*- coding: utf-8 -*-

import config
from ddp_input_feed import create_dataset
from ddp_classifier import EncoderClassifier, TextCNNClassifier
from utils import current_time, cal_avg, restore_sentence_size


import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
"""

import horovod.tensorflow as hvd

# Initialize Horovod
hvd.init()

cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
cfg.gpu_options.allow_growth = True
# Pin GPU to be used to process local rank (one GPU per process)
cfg.gpu_options.visible_device_list = str(hvd.local_rank())


def train():
    print(current_time(), "training starts...")

    # sentence_size_max = restore_sentence_size(config.SENTENCE_FILE)
    sentence_size_max = config.SENTENCE_SIZE_MAX
    print("sentence_size_max=%d" % sentence_size_max)

    g = tf.Graph()
    with g.as_default():
        # create a feed-able iterator, to be feed by train and test datasets
        handle_ph = tf.placeholder(dtype=tf.string, name="handle_ph")
        # create a dataset consists of the worker's shard
        print("num_shards: %d, shard_index: %d" % (hvd.size(), hvd.rank()))
        train_dataset = create_dataset(config.TF_TRAIN_PATH, sentence_size_max, hvd.size(), hvd.rank(), test=False)
        train_iterator = tf.data.make_initializable_iterator(train_dataset)

        """
        if hvd.rank() == 0:
            test_dataset = create_dataset(config.TF_TEST_PATH, sentence_size_max, hvd.size(), hvd.rank(), test=True)
            test_iterator = tf.data.make_initializable_iterator(test_dataset)
        """

        iterator = tf.data.Iterator.from_string_handle(
            handle_ph,
            tf.data.get_output_types(train_dataset),
            tf.data.get_output_shapes(train_dataset),
            tf.data.get_output_classes(train_dataset))
        content, label = iterator.get_next(name="next_batch")

        # create model network
        # To be able to feed with batches of different size, the first dimension should be None
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
        # TODO: it's strange that tf.train.AdagradOptimizer doesn't work here, loss fluctuates
        opt = tf.train.AdamOptimizer(config.LEARNING_RATE * hvd.size())
        # Wrap regular optimizer with Horovod one, which takes care of averaging gradients using ring-allreduce
        dist_opt = hvd.DistributedOptimizer(opt)
        train_op = dist_opt.minimize(loss_op)

        preds_op, acc_op = model.predict(logits, label_ph)

        # create saver
        saver = tf.train.Saver(max_to_keep=1)

    with tf.Session(graph=g, config=cfg) as sess:
        tf.global_variables_initializer().run()
        print("--- global variables")
        for global_variable in tf.global_variables():
            print(global_variable)
        # Broadcast the initial variable states from rank 0 to all other processes/workers.
        #bc_op = hvd.broadcast_variables(opt.variables(), root_rank=0)
        bc_op = hvd.broadcast_global_variables(root_rank=0)
        sess.run(bc_op)

        # create handle to feed to iterator's string_handle placeholder
        train_handle = sess.run(train_iterator.string_handle())
        """
        if hvd.rank() == 0:
            test_handle = sess.run(test_iterator.string_handle())
        """

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

                    if step % config.STEPS_PER_CKPT == 0 and hvd.rank() == 0:
                        print(current_time(), "step: %d, train_loss: %.3f, train_acc: %.3f" % (step, train_loss, train_acc))
                        saver.save(sess, config.CKPT_PATH, global_step=step)

                    step += 1
                except tf.errors.OutOfRangeError:
                    if hvd.rank() == 0:
                        print(current_time(), "step: %d, train_loss: %.3f, train_acc: %.3f" % (step, train_loss, train_acc))
                        saver.save(sess, config.CKPT_PATH, global_step=step)
                    break

            # save model
            if hvd.rank() == 0:
                save_model(sess, config.MODLE_DIR, config.MODEL_NAME)

    print(current_time(), "training finishes...")


def save_model(sess, model_dir, filename):
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        ["logits", "loss", "output/predictions", "output/accuracy"])

    model_filepath = "%s/%s.pb" % (model_dir, filename)
    with tf.gfile.GFile(model_filepath, "wb") as fout:
        fout.write(output_graph_def.SerializeToString())


def main():
    # train
    train()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-

import config
from input_feed import create_dataset
from classifier import DistillEncoderClassifier
from utils import current_time, with_prefix, get_optimizer, load_model, save_model

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


def train_distill():
    print(current_time(), "training starts...")

    print(config.MODEL_NAME)
    if config.MODEL_NAME != "distilled-encoder":
        print("Unsupported model %s" % config.MODEL_NAME)
        exit(-1)

    outputs = ["logits", "loss", "output/predictions", "output/accuracy"]

    # load the teacher model
    tea_sess, tea_content_ph, tea_label_ph, tea_dropout_keep_prob_ph, tea_logits_op = \
        init_teacher_model(config.TEACHER_MODEL_NAME, outputs)

    # sentence_size_max = restore_sentence_size(config.SENTENCE_FILE)
    sentence_size_max = config.SENTENCE_SIZE_MAX
    print("sentence_size_max=%d" % sentence_size_max)

    stu_graph = tf.Graph()
    with stu_graph.as_default():
        # create a feed-able iterator, to be feed by train and test datasets
        handle_ph = tf.placeholder(dtype=tf.string, name="handle_ph")
        train_dataset = create_dataset(config.TF_TRAIN_PATH, sentence_size_max, test=False)
        train_iterator = tf.data.make_initializable_iterator(train_dataset)
        #train_iterator = train_dataset.make_initializable_iterator()

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

        tea_logits_ph = tf.placeholder(dtype=tf.float32, shape=(None, config.NUM_CLASS), name="tea_logits_ph")

        model = DistillEncoderClassifier()
        stu_logits = model.forward(content_ph, dropout_keep_prob_ph)
        loss_op = model.distill_loss(tea_logits_ph, stu_logits, label_ph)
        opt = get_optimizer()
        train_op = opt.minimize(loss_op)
        preds_op, acc_op = model.predict(stu_logits, label_ph)

        # create saver
        saver = tf.train.Saver(max_to_keep=1)

    with tf.Session(graph=stu_graph, config=cfg) as stu_sess:
        tf.global_variables_initializer().run()

        # create handle to feed to iterator's string_handle placeholder
        train_handle = stu_sess.run(train_iterator.string_handle())

        step = 0
        for i in range(config.NUM_EPOCH):
            print(current_time(), "epoch: %d" % (i + 1))
            stu_sess.run(train_iterator.initializer)

            while True:
                try:
                    content_ts, label_ts = stu_sess.run([content, label], feed_dict={handle_ph: train_handle})

                    tea_logits = tea_sess.run(tea_logits_op, feed_dict={tea_content_ph: content_ts,
                                                                        tea_label_ph: label_ts,
                                                                        tea_dropout_keep_prob_ph: config.TEST_KEEP_PROB})

                    _, train_acc, train_loss = stu_sess.run([train_op, acc_op, loss_op],
                                                        feed_dict={content_ph: content_ts,
                                                                   label_ph: label_ts,
                                                                   dropout_keep_prob_ph: config.TRAIN_KEEP_PROB,
                                                                   tea_logits_ph: tea_logits})

                    if step % config.STEPS_PER_CKPT == 0:
                        print(current_time(),
                                "step: %d, train_loss: %.3f, train_acc: %.3f" % (step, train_loss, train_acc))

                        saver.save(stu_sess, config.CKPT_PATH, global_step=step)

                    step += 1
                except tf.errors.OutOfRangeError:
                    print(current_time(),
                          "step: %d, train_loss: %.3f, train_acc: %.3f" % (step, train_loss, train_acc))

                    saver.save(stu_sess, config.CKPT_PATH, global_step=step)
                    break

            # save the distilled model
            save_model(stu_sess, config.MODLE_DIR, config.MODEL_NAME, outputs)

    tea_sess.close()

    print(current_time(), "training finishes...")


def init_teacher_model(model_name, outputs):
    print(current_time(), "init teacher model starts...")

    # load trained model
    tea_graph = load_model(config.MODLE_DIR, model_name, outputs)
    #load_ckpt_model(t_sess, config.CKPT_DIR)
    tea_sess = tf.Session(graph=tea_graph, config=cfg)
    """
    for operation in tea_graph.get_operations():
        print(operation.name)
    """

    tea_content_ph = tea_graph.get_tensor_by_name(with_prefix(model_name, "content_ph:0"))
    tea_label_ph = tea_graph.get_tensor_by_name(with_prefix(model_name, "label_ph:0"))
    tea_dropout_keep_prob_ph = tea_graph.get_tensor_by_name(with_prefix(model_name, "dropout_keep_prob:0"))
    tea_logits_op = tea_graph.get_tensor_by_name(with_prefix(model_name, "logits:0"))

    print(current_time(), "init teacher model finished!")

    return tea_sess, tea_content_ph, tea_label_ph, tea_dropout_keep_prob_ph, tea_logits_op


def main():
    # train the distilled model
    train_distill()


if __name__ == "__main__":
    main()

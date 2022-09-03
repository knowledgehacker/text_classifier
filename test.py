# -*- coding: utf-8 -*-
import os

import config
from input_feed import create_dataset
from utils import current_time, cal_avg

"""
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
"""

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
cfg.gpu_options.allow_growth = True


def test():
    print(current_time(), "testing starts...")

    #sentence_size_max = restore(config.SENTENCE_FILE)
    sentence_size_max = config.SENTENCE_SIZE_MAX
    print("sentence_size_max=%d" % sentence_size_max)

    g = tf.Graph()
    with tf.Session(graph=g, config=cfg) as sess:
        # load trained model
        load_model(sess, config.CKPT_DIR)

        # get prediction and other dependent tensors from the graph in the trained model for inference
        content_ph = g.get_tensor_by_name("content_ph:0")
        label_ph = g.get_tensor_by_name("label_ph:0")
        dropout_keep_prob_ph = g.get_tensor_by_name("dropout_keep_prob:0")
        preds_op = g.get_tensor_by_name("output/predictions:0")
        acc_op = g.get_tensor_by_name("output/accuracy:0")

        # create iterator for test dataset
        handle_ph = g.get_tensor_by_name("handle_ph:0")
        test_dataset = create_dataset(config.TF_TEST_PATH, sentence_size_max, test=True)
        test_iterator = test_dataset.make_initializable_iterator()

        test_handle = sess.run(test_iterator.string_handle())

        # important!!! Don't call 'tf.global_variables_initializer().run()' when doing inference using trained model
        #tf.global_variables_initializer().run()
        sess.run(test_iterator.initializer)

        all_test_labels = []
        all_test_preds = []

        accs = []
        try:
            while True:
                # get next batch from test dataset iterator, feed to 'get_next' operator in the graph loaded from trained model
                #content, label = sess.run(get_next_op)
                content, label = test_iterator.get_next("next_batch")
                content_ts, label_ts = sess.run([content, label], feed_dict={handle_ph: test_handle})
                all_test_labels.append(label_ts)

                preds, acc = sess.run([preds_op, acc_op], feed_dict={content_ph: content_ts,
                                                                     label_ph: label_ts,
                                                                     dropout_keep_prob_ph: config.TEST_KEEP_PROB})
                all_test_preds.append(preds)

                accs.append((acc, len(preds)))

                print('acc: %.3f' % acc)
        except tf.errors.OutOfRangeError:
            print('acc: %.3f' % acc)
            pass

        """
        with open("data/preds.txt", 'w', encoding='utf-8') as pf:
            pf.write("label\t\preds\n")
            for i in range(len(all_test_labels)):
                pf.write("%s\t\%s" % (all_test_labels[i], all_test_preds))
        """

        avg_acc = cal_avg(accs)
        print("avg_acc: %.3f" % avg_acc)

    print(current_time(), "testing finishes...")


def load_model(sess, ckpt_dir):
    ckpt_file = tf.train.latest_checkpoint(ckpt_dir)
    print("ckpt_file: %s" % ckpt_file)
    saver = tf.train.import_meta_graph("{}.meta".format(ckpt_file))
    saver.restore(sess, ckpt_file)


def main():
    # test
    test()


if __name__ == "__main__":
    main()

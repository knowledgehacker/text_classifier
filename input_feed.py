# -*- coding: utf-8 -*-

import config

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


def create_dataset(input, sentence_size_max, test):
    dataset = tf.data.TFRecordDataset(input)

    dataset = dataset.map(parse)
    dataset = dataset.map(lambda content, label: (content, tf.size(content), label))
    dataset = dataset.filter(lambda content, size, label:
                             tf.logical_and(tf.greater(size, config.SENTENCE_SIZE_MIN),
                                            tf.less_equal(size, sentence_size_max)))
    dataset = dataset.map(lambda content, size, label: (content, label))

    # TODO: how padding here affects training?
    """
    Padding with 0, which is the same as the index of config.PAD,
    will this cause train result incorrect?
    Considering the case max pooling after convolutions selects index 0, which is the padding 0,
    will the model learns the padding 0 is insignificant after convolutions?
    """
    with tf.device('/cpu:0'):
        padded_shapes = (
            (tf.TensorShape([sentence_size_max]),
             tf.TensorShape([config.NUM_CLASS])))
        if not test:
            dataset = dataset.shuffle(config.SHUFFLE_SIZE)
            dataset = dataset.padded_batch(config.BATCH_SIZE, padded_shapes)
        else:
            dataset = dataset.padded_batch(config.TEST_BATCH_SIZE, padded_shapes)

    return dataset


def parse(record):
    example = tf.io.parse_single_example(
        record,
        features={
            'indices': tf.VarLenFeature(tf.int64),
            'label': tf.FixedLenFeature([config.NUM_CLASS], tf.float32)
        })

    # represent 'indices' as a dense tensor, used as a feed to tf.nn.embedding_lookup. refer to seq2seq model
    return tf.sparse_tensor_to_dense(example['indices']), example['label']

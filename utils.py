# -*- coding: utf-8 -*-

import time
import os

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

import config


def current_time():
    return time.strftime('%H:%M:%S', time.localtime(time.time()))


def load_files(input):
    files = []
    for dir in os.listdir(input):
        path = os.path.join(input, dir)
        if os.path.isdir(path):
            for file in os.listdir(path):
                if not file.endswith("_SUCCESS"):
                    files.append(os.path.join(path, file))
        else:
            if not path.endswith("_SUCCESS"):
                files.append(path)

    return files


def load_model(model_dir, model_name, outputs):
    model_filepath = "%s/%s.pb" % (model_dir, model_name)

    print("Loading model %s ..." % model_filepath)

    with tf.gfile.FastGFile(model_filepath, 'rb') as fin:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fin.read())

    g = tf.Graph()
    with g.as_default():
        tf.import_graph_def(graph_def, return_elements=outputs, name=model_name)

    print("Model %s loaded!" % model_filepath)

    return g


def save_model(sess, model_dir, model_name, outputs):
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        outputs)

    model_filepath = "%s/%s.pb" % (model_dir, model_name)
    with tf.gfile.GFile(model_filepath, "wb") as fout:
        fout.write(output_graph_def.SerializeToString())


def get_optimizer():
    if config.OPTIMIZER == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=config.LEARNING_RATE)
    elif config.OPTIMIZER == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=config.LEARNING_RATE, initial_accumulator_value=1e-8)
    else:
        print("Unsupported optimizer: %s" % config.OPTIMIZER)

    return optimizer


def with_prefix(prefix, op):
    return "%s/%s" % (prefix, op)
    #return op


def is_punctuation(char):
    if char == ',' \
            or char == '.'  \
            or char == ':'  \
            or char == ';'  \
            or char == '!'  \
            or char == '?' \
            or char == '\'' \
            or char == '"':
        return True

    return False


def load_vocab(vocab_file):
    vocab = dict()

    with open(vocab_file, mode='r', encoding='utf-8') as vf:
        lines = vf.readlines()
        for i in range(len(lines)):
            line = lines[i].strip()
            vocab[line] = i

    return vocab


def save_vocab(words, vocab_file):
    with open(vocab_file, mode='w', encoding='utf-8') as vf:
        for word in words:
            vf.write("%s\n" % word)


def save_sentence_size(sentence_file, sentence_max_size):
    with open(sentence_file, 'w') as fout:
        fout.write(str(sentence_max_size))


def restore_sentence_size(sentence_file):
    with open(sentence_file, 'r') as fin:
        sentence_max_size = int(fin.readline().strip())

    return sentence_max_size


def cal_avg(values):
    total = 0.0
    total_num = 0
    for value, num in values:
        total += value * num
        total_num += num
    #print("total_num: %d" % total_num)
    avg = total / total_num

    return avg

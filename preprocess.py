# -*- coding: utf-8 -*-

import collections
from operator import itemgetter

import config
from utils import current_time, save_vocab, load_vocab, save_sentence_size

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


# Build vocabulary
def build_vocab(input, vocab_file, vocab_size):
    print(current_time(), "build vocabulary: %s starts..." % vocab_file)

    wc = collections.Counter()
    with open(input, 'r', encoding='utf-8') as f:
        for line in f:
            splits = line.strip().split('\t')
            if len(splits) == 2:
                content = list(splits[1])   # split a sentence into words
                for word in content:
                    wc[word] += 1

    sorted_wc = sorted(wc.items(), key=itemgetter(1), reverse=True)
    sorted_words = [w for w, c in sorted_wc]

    # handle special words
    sorted_words_with_special = ['[PAD]'] \
                                + sorted_words[:99] \
                                + ['[UNK]', '[CLS]', '[SEP]'] \
                                + sorted_words[99:]
    if len(sorted_words_with_special) > vocab_size:
        sorted_words_with_special = sorted_words_with_special[:vocab_size]

    save_vocab(sorted_words_with_special, vocab_file)

    print(current_time(), "build vocabulary: %s finishes..." % vocab_file)


# Build dataset, replace words with indices, with rare words replaced with '[UNK]'
def build_dataset(input, vocab_file, output):
    print(current_time(), "build dataset: %s starts..." % input)

    cate_to_index = build_cate_to_index()

    vocab = load_vocab(vocab_file)

    sample_num = 0

    sentence_max_size = -1

    fin = open(input, 'r', encoding='utf-8')
    writer = tf.python_io.TFRecordWriter(output)
    for line in fin:
        splits = line.strip().split('\t')
        if len(splits) == 2:
            content = list(splits[1])

            content_size = len(content)
            if content_size > sentence_max_size:
                sentence_max_size = content_size

            if config.SENTENCE_SIZE_MIN < content_size <= config.SENTENCE_SIZE_MAX:
                sample_num += 1

            indices = [get_index('[CLS]', vocab)] \
                      + [get_index(word, vocab) for word in content] \
                      + [get_index('[SEP]', vocab)]
            label = one_hot_encode(cate_to_index[splits[0]], config.NUM_CLASS)
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'indices': int64_feature(indices),
                    'label': float_feature(label)
                }))
            writer.write(example.SerializeToString())
    fin.close()
    writer.close()

    print("sentence_max_size=%d" % sentence_max_size)

    print("sample_num=%d" % sample_num)

    print(current_time(), "build dataset: %s finishes..." % input)

    return sentence_max_size


def build_cate_to_index():
    news_cates = [news_category for news_category in config.NEWS_CATEGORIES]  # 'utf-8'
    cate_to_index = dict(zip(news_cates, range(config.NUM_CLASS)))

    return cate_to_index


def get_index(word, word_index):
    return word_index[word] if word in word_index else word_index[config.SYM_UNK]


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def one_hot_encode(index, num):
    ohe = [0.0 for i in range(num)]
    ohe[index] = 1.0

    return ohe


build_vocab(config.RAW_TRAIN_PATH, config.VOCAB_FILE, config.VOCAB_SIZE)
train_sentence_max_size = build_dataset(config.RAW_TRAIN_PATH, config.VOCAB_FILE, config.TF_TRAIN_PATH)
save_sentence_size(config.SENTENCE_FILE, train_sentence_max_size)
build_dataset(config.RAW_TEST_PATH, config.VOCAB_FILE, config.TF_TEST_PATH)


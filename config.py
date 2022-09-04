MODEL_NAME = 'transformer'

CKPT_DIR = 'ckpt'
#MODEL_DIR = 'models'

DATA_DIR = 'data'
TRAIN_PATH = '%s/train' % DATA_DIR
TEST_PATH = '%s/test' % DATA_DIR

RAW_TRAIN_PATH = '%s/cnews.train.txt' % TRAIN_PATH
RAW_TEST_PATH = '%s/cnews.test.txt' % TEST_PATH
"""
RAW_TRAIN_PATH = 'data/x.cnews.train.txt'
RAW_TEST_PATH = 'data/cnews.test.txt'
"""
TF_TRAIN_PATH = '%s/cnews.train.tfrecords' % TRAIN_PATH
TF_TEST_PATH = '%s/cnews.test.tfrecords' % TEST_PATH

SYM_PAD = '[PAD]'
SYM_UNK = '[UNK]'
SYM_CLS = '[CLS]'
SYM_SEP = '[SEP]'
SYM_MASK = '[MASK]'

VOCAB_SIZE = 50000
VOCAB_FILE = '%s/vocab.zh' % DATA_DIR


SENTENCE_FILE = 'data/sentence.txt'

SENTENCE_SIZE_MIN = 1
# documents: 50000 - 2500 length, 47391 remains; 1000 length, 34331 remains
SENTENCE_SIZE_MAX = 1000
#SENTENCE_SIZE_MAX = 600

NUM_EPOCH = 5

# shuffle size affects convergence greatly, it should be big enough
SHUFFLE_SIZE = 5000

# large batch, ex 200, does not work, I don't know why
BATCH_SIZE = 128

STEPS_PER_CKPT = 50

TEST_BATCH_SIZE = 1000

VALIDATE = False


LEARNING_RATE = 1e-3

TRAIN_KEEP_PROB = 0.7
TEST_KEEP_PROB = 1.0

"""
EMBED_SIZE = 768
NUM_ATTENTION_HEAD = 12
"""
EMBED_SIZE = 144
NUM_ATTENTION_HEAD = 3

NUM_ENCODER_LAYER = 3


# news categories
NEWS_CATEGORIES = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
#                   0       1       2       3       4       5       6   7          8    9

NUM_CLASS = len(NEWS_CATEGORIES)




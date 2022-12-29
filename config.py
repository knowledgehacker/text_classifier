MODEL_NAME = 'distilled-encoder'
#MODEL_NAME = 'encoder'
#MODEL_NAME = 'textcnn'

MODLE_DIR = "models"

CKPT_DIR = 'ckpt'
CKPT_PATH = '%s/%s' % (CKPT_DIR, MODEL_NAME)

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

STEPS_PER_CKPT = 100

VALIDATE = False

# shuffle size affects convergence greatly, it should be big enough
SHUFFLE_SIZE = 5000

# large batch, ex 200, does not work, I don't know why
BATCH_SIZE = 128
TEST_BATCH_SIZE = 300

OPTIMIZER = 'adam'

LEARNING_RATE = 1e-3

TRAIN_KEEP_PROB = 0.7
TEST_KEEP_PROB = 1.0

if MODEL_NAME == 'textcnn':
    EMBED_SIZE = 64

    HIDDEN_SIZE = 64

    CONV_FILTER_NUM = 128
    CONV_FILTER_KERNEL_SIZES = [2, 3, 4]
elif MODEL_NAME == 'encoder':
    # EMBED_SIZE = 768
    EMBED_SIZE = 144

    # NUM_ATTENTION_HEAD = 12
    NUM_ATTENTION_HEAD = 4

    NUM_ENCODER_LAYER = 4
elif MODEL_NAME == 'distilled-encoder':
    TEACHER_MODEL_NAME = 'encoder'

    # EMBED_SIZE = 768
    EMBED_SIZE = 144

    # NUM_ATTENTION_HEAD = 12
    NUM_ATTENTION_HEAD = 2

    NUM_ENCODER_LAYER = 2

    T = 2.0
    alpha = 0.5
else:
    print("Unsupported model %s" % MODEL_NAME)
    exit(-1)

# news categories
NEWS_CATEGORIES = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
#                   0       1       2       3       4       5       6   7          8    9

NUM_CLASS = len(NEWS_CATEGORIES)

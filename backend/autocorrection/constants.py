import re

# PREPROCESS
TOKENIZER_REGEX = re.compile(r'(\W)')

LEGAL = " !\"#$%&'()*+,-./0123456789:;<=>?@[\\]^_`abcdefghijklmnopqrstuvwxyzáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ{|}~"
PUNCT = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

# TOKENIZATION_REPAIR
APPROACH = "CUSTOM"
P_INS = 4
P_DEL = 4
PENALTIES = ""
FWD = "lm/unilm"
BID = None

DEVICE = "cpu"

# Continue train with the previous model
IS_CONTINUOUS_TRAIN=False

PATH_PRETRAINED_MODEL='autocorrection/weights/model/model_last.pth'

# Using the default split data for training and validation
IS_SPLIT_INDEXES=True

# Number epoches
N_EPOCH=100

# lambda for control the important level of detection loss value
LAMDA=0.6

# parameter controlled the contribution of the label 0 in the detection loss
PENALTY_VALUE=0.3

USE_DETECTION_CONTEXT=True

# For preprocessing
BERT_PRETRAINED = 'vinai/phobert-base'
MODEL_SAVE_PATH = 'autocorrection/weights/model/'
MODEL_CHECKPOINT = f'model_final.pth'

PKL_PATH = 'autocorrection/input/luanvan/'

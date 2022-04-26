import re

# PREPROCESS
TOKENIZER_REGEX = re.compile(r'(\W)')

LEGAL = " !\"#$%&'()*+,-./0123456789:;<=>?@[\\]^_`abcdefghijklmnopqrstuvwxyzáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ{|}~"
PUNCT = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

# TOKENIZATION_REPAIR
APPROACH = "CUSTOM"
P_INS = 0
P_DEL = 0
PENALTIES = ""
FWD = "lm/unilm"
BID = None
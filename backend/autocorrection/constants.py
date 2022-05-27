import re

# PREPROCESS
TOKENIZER_REGEX = re.compile(r'(\W)')

LEGAL = " !\"#$%&'()*+,-./0123456789:;<=>?@[\\]^_`abcdefghijklmnopqrstuvwxyzáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ{|}~"
PUNCT = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

# TOKENIZATION_REPAIR
APPROACH = "CUSTOM"
P_INS = 6
P_DEL = 6
PENALTIES = ""
FWD = "lm/unilm"
BID = None
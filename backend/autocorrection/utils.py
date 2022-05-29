import re
import pickle
import unicodedata
from collections import defaultdict
from underthesea import word_tokenize

from autocorrection.constants import *


def reverse_tok_special(sentence):
    numberic_match = r'\bn\s*u\s*m\s*b\s*e\s*r\s*i\s*c\b'
    date_match = r'\bd\s*a\s*t\s*e\b'
    sentence = re.sub(numberic_match, r'numberic', sentence)
    sentence = re.sub(date_match, r'date', sentence)
    return sentence


def tokenize(text):
    tokens = TOKENIZER_REGEX.split(text)
    return [t for t in tokens if len(t.strip()) > 0]


def load_pickle_file(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def save_pickle_file(path, data):
    with open(path, 'wb') as file:
        pickle.dump(data, file, protocol=4)


def preprocess(sent, remove_punct=True):
    # if remove_punct:
    #     sent = re.sub('[,.!?|]', " ", sent)
    sent = separate_number_chars(sent)
    sent = unicodedata.normalize("NFC", sent)
    sent = re.sub(r'[\t\n]', ' ', sent)
    sent = re.sub(r'\s+', r' ', sent)
    sent = re.sub(r'^\s', '', sent)
    sent = re.sub(r'\s$', '', sent)
    sent = sent.rstrip('_+!@#$?^')
    sent = sent.lstrip('_+!@#$?^')
    return sent


def processing_after(sent):
    sent = re.sub('\n', '', sent)
    sent = re.sub(r'\s+', r' ', sent)
    sent = re.sub(r'^\s', '', sent)
    sent = re.sub(r'\s$', '', sent)
    sent = re.sub(r'\s\.\s', '.', sent)
    sent = re.sub(r'\s/\s', '/', sent)
    sent = re.sub(r'\s+(?=(\.|,))', '', sent)
    sent = re.sub(r'\s+/\s+', '/', sent)
    sent = re.sub(r'\s,\s', ', ', sent)
    sent = re.sub(r'\s.\s', '. ', sent)
    return sent


def separate_number_chars(s):
    res = re.split('\b([-+]?\d+\.\d+)|([-+]?\d+)\b', s.strip())
    res_f = [r.strip() for r in res if r is not None and r.strip() != '']
    return ' '.join(res_f)


def match_case(src, predicted):
    src = src.strip()
    out = []
    if len(predicted.split()) == len(src.split()):
        for i in range(len(predicted)):
            if src[i].isupper():
                out.append(predicted[i].upper())
            elif src[i] in PUNCT or src[i] not in LEGAL:
                out.append(src[i])
            else:
                out.append(predicted[i])
        return ''.join(out)
    else:
        return predicted


def create_label(text):

    '''
    Take a string -> intext and label
    '''
    tokens = word_tokenize(text)
    words = []
    ids_punct = {',':[], '.':[]}
    i = 0
    for token in tokens:
        if token not in ids_punct.keys():
            words.append(token)
            i+=1
        else:
            ids_punct[token].append(i-1)

    label = [0]*len(words)
    for pun, ids in ids_punct.items():
        for index in ids:
            label[index] = 1 if pun == ',' else 2
    return label


def match_punct(src_sent, predicted_sent):
    if len(predicted_sent.split()) == len(src_sent.split()):
        label = create_label(predicted_sent)
        words = word_tokenize(src_sent)
        
        convert = {0: '', 1: ',', 2: '.', 3: ''}
        seq = [ word+convert[label[i]] for i, word in enumerate(words)]
        seq = ' '.join(seq)
    
        return '. '.join(map(lambda s: s.strip().capitalize(), seq.split('.')))
    else:
        return src_sent


def is_number(token):
    if token.isnumeric():
        return True
    return bool(re.match('(\d+[\.,])+\d', token))


def is_date(token):
    return bool(re.match('(\d+[-.\/])+\d+', token))


def mark_special_token(sentence):
    tokens = word_tokenize(sentence)
    index_special = defaultdict(list)
    for i in range(len(tokens)):
        # if is_number(tokens[i]):
        #     index_special['numberic'].append(tokens[i])
        #     tokens[i] = 'numberic'
        if is_date(tokens[i]):
            index_special['date'].append(tokens[i])
            tokens[i] = 'date'
        # elif is_special_token(tokens[i]):
        #     index_special['specialw'].append(tokens[i])  # mark differ 'special' word
        #     tokens[i] = 'specialw'
    return " ".join(tokens), index_special


def split_token(tokenizer, sentence: str):
    """
    Calculate number token for each word
    Example:
    tokenize("Xiin chàoo ngày mới") ==> ['Xi@@', 'in', 'ch@@', 'à@@', 'oo', 'ngày', 'mới']
    ==> result: [2,3,1,1]
    """

    list_words = sentence.split()
    tokens = tokenizer.tokenize(sentence)
    start = 0
    result = []
    size = len(list_words)
    for i in range(size):
        word = list_words[i]
        if i < size - 1:
            str_temp = ""
            start_temp = start
            while start < len(tokens) and str_temp != word:
                index = tokens[start].find('@')
                str_temp += tokens[start][:index] if index != -1 else tokens[start]
                start += 1
            result.append(start - start_temp)

        else:
            result.append(len(tokens) - start)
    assert sum(result) == len(tokens), "Number tokens not equal"
    return result
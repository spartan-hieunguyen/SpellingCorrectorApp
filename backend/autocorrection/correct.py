import re
import string
from itertools import groupby

import numpy as np
import nltk
nltk.download("punkt")
from nltk import sent_tokenize, word_tokenize
from transformers import AutoTokenizer

from autocorrection.utils import *
from autocorrection.model.model import *
from autocorrection.params import *
from autocorrection.vietnamese_normalizer import VietnameseNormalizer
from autocorrection.postprocess import RuleBasedPostprocessor
from autocorrection.get_corrector import get_corrector
from autocorrection.test import perfect_align

class AutoCorrection:
    def __init__(self, threshold_detection=0.7, threshold_correction=0.6, use_detection_context=False):
        self.device = DEVICE
        self.model_name = "phobert"
        self.use_detection_context = use_detection_context
        self.threshold_correction = threshold_correction
        self.threshold_detection = threshold_detection
        self.word_tokenizer = load_pickle_file(PKL_PATH + 'word_tokenizer_we.pkl')
        self.phobert_tokenizer = AutoTokenizer.from_pretrained(BERT_PRETRAINED)
        self.path_pretrained = MODEL_SAVE_PATH + MODEL_CHECKPOINT
        self.model = self.load_model(self.path_pretrained)
        self.normalizer = VietnameseNormalizer()
        self.postprocessor = RuleBasedPostprocessor()
        self.tokenization_repair = get_corrector(approach=APPROACH,
                                                 penalties=PENALTIES,
                                                 insertion_penalty=P_INS,
                                                 deletion_penalty=P_DEL,
                                                 fwd_model_name=FWD,
                                                 bid_model_name=BID)

    def select_model(self):
        model = PhoBertEncoder(n_words=40000,
                                n_labels_error=2,
                                use_detection_context = self.use_detection_context
                                ).to(self.device)
        return model

    def load_model(self, path):
        model = self.select_model()
        model_states = torch.load(path, map_location=self.device)
        model.load_state_dict(model_states['model'])
        model.eval()
        return model

    def make_inputs(self, sentence):
        data = []
        word_ids = self.phobert_tokenizer.encode(sentence)
        data = torch.tensor(word_ids, dtype=torch.long).to(self.device).unsqueeze(dim=0)
        batch_ids = batch_ids = [split_token(self.phobert_tokenizer, sentence)]
        return data, None, batch_ids

    def restore_sentence(self, sentence, mark_replaces):
        tokens = word_tokenize(sentence)
        start_1, start_2, start_3 = 0, 0, 0
        for i in range(len(tokens)):
            if tokens[i] == 'numberic' and start_1 < len(mark_replaces['numberic']):
                tokens[i] = mark_replaces['numberic'][start_1]
                start_1 += 1
            elif tokens[i] == 'date' and start_2 < len(mark_replaces['date']):
                tokens[i] = mark_replaces['date'][start_2]
                start_2 += 1
            elif tokens[i] == 'specialw' and start_3 < len(mark_replaces['specialw']):
                tokens[i] = mark_replaces['specialw'][start_3]
                start_3 += 1
        sentence = " ".join(tokens)
        return sentence

    def argmax_tensor(self, detection_outputs, correction_outputs):
        detection_prob, detection_indexs = torch.max(detection_outputs, dim=-1)
        correction_prob, correction_indexs = torch.max(correction_outputs, dim=1)

        return detection_prob.detach().cpu().numpy(), \
               detection_indexs.detach().cpu().numpy(), \
               correction_prob.detach().cpu().numpy(), \
               correction_indexs.detach().cpu().numpy()

    def get_result(self, convert_word, sentence, detection_outputs, correction_outputs):
        words = sentence.split()
        detection_prob, detection_indexs, correction_prob, correction_indexs = \
            self.argmax_tensor(detection_outputs, correction_outputs)
        for index, value in enumerate(detection_prob):
            # if the probability for not the spell word is less then threshold, it's is spell word
            if value < self.threshold_detection and detection_indexs[index] == 0:
                detection_indexs[index] = 1

            if index in convert_word.keys():
                detection_indexs[index] = 1
                correction_indexs[index] = self.word_tokenizer.texts_to_sequences([convert_word[index]])[0][0]

        wrong_word_indexs = np.where(detection_indexs == 1)[0]
        word_predict = correction_indexs[wrong_word_indexs]
        word_predict = self.word_tokenizer.sequences_to_texts([word_predict])[0].split()
        if len(wrong_word_indexs) > 0:
            for index1, index2 in zip(wrong_word_indexs, range(len(word_predict))):
                if words[index1] == word_predict[index2]:
                    detection_indexs[index1] = 0
                    continue
                # if a word is out of vocabulary, then the probability for word prediction need greater than a
                # threshold,else predict with normal
                if correction_prob[index1] > self.threshold_correction:
                    words[index1] = word_predict[index2]
                else:
                    detection_indexs[index1] = 0
                    
        # assert len(words) == len(detection_indexs) == len(correction_indexs)
        return words, detection_indexs.tolist(), correction_indexs.tolist()

    def forward(self, original_sentence):
        convert_word = {}
        words = original_sentence.split()
        for idx, word in enumerate(words):
            word_norm = self.normalizer.normalize(word)
            if word != word_norm:
                convert_word[idx] = word_norm
                words[idx] = word_norm
        original_sentence = " ".join(words)
        data = self.make_inputs(original_sentence)
        detection_outputs, correction_outputs = self.model(*data)
        detection_outputs, correction_outputs = torch.softmax(detection_outputs, dim=-1), torch.softmax(
            correction_outputs, dim=-1)
        words, detection_predict, correction_predict = self.get_result(convert_word, original_sentence,
                                                                          detection_outputs.squeeze(dim=0),
                                                                          correction_outputs.squeeze(dim=0))

        return words, detection_predict, correction_predict

    def normalize(self, sentence):
        sentence = re.sub('[,.!?|]', " ", sentence)
        sentence = separate_number_chars(sentence)
        sentence = " ".join(sentence.split())
        
        return sentence
    
    def _match_case(self, original_sentence, predicted, data):
        tokens = predicted.split()
        itokens = tokenize(original_sentence)
        #rematch case
        cased_tokens=[]
        assert len(tokens) == len(data["case"])
        for tok, case, ori in zip(tokens, data["case"], itokens):
            if tok == '<unk>':
              cased_tokens.append(ori)
            elif tok == '<del>':
              cased_tokens.append('')
            elif ori.lower() == tok:
              cased_tokens.append(ori)
            elif case == 'O':
              cased_tokens.append(tok)
            elif case == 'C':
              cased_tokens.append(tok.capitalize())
            else:
              cased_tokens.append(tok.upper())
        return self._reverse_tokenizer(cased_tokens, data["gap"])

    def _reverse_tokenizer(self,tokens, gaps):
        assert 0 <= len(gaps) - len(tokens) <=1
        if len(gaps) > len(tokens): tokens.append('')
        ans = ''
        for t, g in zip(tokens, gaps):
            ans = ans + g + t
        return ans
        
    def _findAllGap(self, text, tokens):
        assert text.startswith(tokens[0])
        gaps = []
        gap = ''
        while text or tokens:
            if not tokens:
                gap = text
                gaps.append(gap)
                break
            if text.startswith(tokens[0]):
                text = text[len(tokens[0]):]
                tokens = tokens[1:]
                gaps.append(gap)
                gap = ''
            else:
                gap += text[0]
                text = text[1:]
        return gaps    

    def _preprocess_sentences(self, sents, mode):
        sentences = sent_tokenize(sents)
        result = []
        for s in sentences:
            if mode in [0, 2]:
                s = self.tokenization_repair.correct(s)
            tokens = tokenize(s)
            data = {}
            data["gap"] = self._findAllGap(s, tokens)
                        
            # U: uppercase, C: capitalize, O: nothing
            cases = []
            for t in tokens:
              if t.isupper():
                cases.append("U")
              elif t[0].isupper():
                cases.append("C")
              else: cases.append("O")
            data["case"] = cases

            # ready = ' '.join([t.lower() for t in tokens])
            result.append([' '.join(tokens), data])
        return result

    def _splitWithIndices(self, s, c=' '):
        p = 0
        for k, g in groupby(s, lambda x:x==c):
            q = p + sum(1 for i in g)
            if not k:
                yield p, q
            p = q

    def _process_detection(self, words, detection_output):
        assert len(words) == len(detection_output)
        
        output = []
        for w, d in zip(words, detection_output):
            if w in string.punctuation:
                continue
            output.append(d)
        return output
          
    def _concat_spans(self, spans):
        sorted_spans = sorted(spans, key=lambda d: d['start']) 
        results = [sorted_spans[0]]
        for span in sorted_spans[1:]:
            if span["start"] <= results[-1]["end"]:
                results[-1]["end"] = span["end"]
            else:
                results.append(span)
        return results
            
    def correction(self, input_sentences, mode):
        results = ""
        spans = []
        detection_outputs = []
        
        input_sentences = preprocess(input_sentences)        
        pairs = self._preprocess_sentences(input_sentences, mode)
        
        repaired_sentences = " ".join([pair[0] for pair in pairs])
        
        if mode == 0:
            _, char_aligns = perfect_align(repaired_sentences, input_sentences)
            for src, trg, idx in char_aligns:
                src = src[len("CHANGE_"):]
                if src == " " and trg == "":
                    start, end = idx
                    for i in range(idx[1] - 1, -1, -1):
                        if repaired_sentences[i] != " ":
                            start -= 1
                            
                    spans.append({
                        "start": start,
                        "end": end
                    })
            spans = self._concat_spans(spans)

            return repaired_sentences, spans
        else:
            for original_sentence, data in pairs:
                words, detection_predict, _ = self.forward(original_sentence.lower())
                detection_predict = self._process_detection(words, detection_predict)
                detection_outputs.extend(detection_predict)
                
                results = results + ' ' + self._match_case(original_sentence, " ".join(words), data)
                results = self.postprocessor.correct(results)

            results = results.strip()

            span_splits = list(self._splitWithIndices(results))
            detection_outputs = np.array(detection_outputs)
            error_idx = np.where(detection_outputs==1)[0]

            _, char_aligns = perfect_align(results, input_sentences)

            for idx in error_idx:
                start, end = span_splits[idx]
                word = results[start:end]
                if word[0] in string.punctuation and word[-1] in string.punctuation:
                    start += 1
                    end -=1
                
                spans.append({
                    "start": start,
                    "end": end
                })
            # for src, trg, idx in char_aligns:
            #     src = src[len("CHANGE_"):]
            #     if src != trg and src not in string.punctuation:
            #         spans.append({
            #             "start": idx[0],
            #             "end": idx[1]
            #         })
            
            # for span in spans:
            #     print(results[span["start"]:span["end"]])
            spans = self._concat_spans(spans)
            return results.strip(), spans
import re
import string

import numpy as np
import nltk

nltk.download("punkt")
from nltk import sent_tokenize, word_tokenize
from transformers import AutoTokenizer

from autocorrection.utils import *
from autocorrection.model.model import *
from autocorrection.vietnamese_normalizer import VietnameseNormalizer
from autocorrection.postprocess import RuleBasedPostprocessor
from autocorrection.get_corrector import get_corrector
from autocorrection.align_sentence import align_sentence


class AutoCorrection:
    def __init__(self, 
                 threshold_detection=0.7, 
                 threshold_correction=0.6, 
                 use_detection_context=False,
                 p_ins=P_INS,
                 p_del=P_DEL):
        self.device = DEVICE
        self.use_detection_context = use_detection_context
        self.threshold_correction = threshold_correction
        self.threshold_detection = threshold_detection
        self.word_tokenizer = load_pickle_file(PKL_PATH + 'word_tokenizer_we.pkl')
        self.phobert_tokenizer = AutoTokenizer.from_pretrained(BERT_PRETRAINED)
        self.path_pretrained = MODEL_SAVE_PATH + MODEL_CHECKPOINT
        self.model = self._load_model(self.path_pretrained)
        self.normalizer = VietnameseNormalizer()
        self.postprocessor = RuleBasedPostprocessor()
        self.p_ins = p_ins
        self.p_del = p_del
        self.tokenization_repair = get_corrector(insertion_penalty=self.p_ins,
                                                 deletion_penalty=self.p_del,
                                                 fwd_model_name=FWD,
                                                 bid_model_name=BID)

    def _select_model(self):
        model = PhoBertEncoder(n_words=40000,
                                n_labels_error=2,
                                use_detection_context = self.use_detection_context
                                ).to(self.device)
        return model

    def _load_model(self, path):
        model = self._select_model()
        model_states = torch.load(path, map_location=self.device)
        model.load_state_dict(model_states['model'])
        model.eval()
        return model

    def _make_inputs(self, sentence):
        data = []
        word_ids = self.phobert_tokenizer.encode(sentence)
        data = torch.tensor(word_ids, dtype=torch.long).to(self.device).unsqueeze(dim=0)
        batch_ids = batch_ids = [split_token(self.phobert_tokenizer, sentence)]
        return data, None, batch_ids

    def _restore_sentence(self, sentence, mark_replaces):
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

    def _argmax_tensor(self, detection_outputs, correction_outputs):
        detection_prob, detection_indexs = torch.max(detection_outputs, dim=-1)
        correction_prob, correction_indexs = torch.max(correction_outputs, dim=1)

        return detection_prob.detach().cpu().numpy(), \
               detection_indexs.detach().cpu().numpy(), \
               correction_prob.detach().cpu().numpy(), \
               correction_indexs.detach().cpu().numpy()

    def _get_result(self, convert_word, sentence, detection_outputs, correction_outputs):
        words = sentence.split()
        detection_prob, detection_indexs, correction_prob, correction_indexs = \
            self._argmax_tensor(detection_outputs, correction_outputs)
            
        for d_i, d_p, c_i, c_p in zip(detection_indexs, detection_prob, correction_indexs, correction_prob):
            word = self.word_tokenizer.sequences_to_texts([[c_i]])[0]
            print(f"{d_i}\t{d_p}\t{word}\t{c_p}")

        for index, _ in enumerate(detection_prob):
            if index in convert_word.keys() and correction_prob[index] > self.threshold_correction:
                detection_indexs[index] = 1             
                
        wrong_word_indexs = np.where(detection_indexs == 1)[0]
        word_predict = correction_indexs[wrong_word_indexs]
        word_predict = self.word_tokenizer.sequences_to_texts([word_predict])[0].split()
        if len(wrong_word_indexs) > 0:
            for index1, index2 in zip(wrong_word_indexs, range(len(word_predict))):
                if correction_prob[index1] > self.threshold_correction:
                    words[index1] = word_predict[index2]
                    
        return words, detection_indexs.tolist(), correction_indexs.tolist()

    def _forward(self, original_sentence):
        convert_word = {}
        words = original_sentence.split()
        for idx, word in enumerate(words):
            word_norm = self.normalizer.normalize(word)
            if word != word_norm:
                convert_word[idx] = word_norm
                words[idx] = word_norm
        original_sentence = " ".join(words)
        data = self._make_inputs(original_sentence)
        detection_outputs, correction_outputs = self.model(*data)
        detection_outputs, correction_outputs = torch.softmax(detection_outputs, dim=-1), torch.softmax(
            correction_outputs, dim=-1)
        words, detection_predict, correction_predict = self._get_result(convert_word, original_sentence,
                                                                          detection_outputs.squeeze(dim=0),
                                                                          correction_outputs.squeeze(dim=0))

        return words, detection_predict, correction_predict
    
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
            sentence, mark_replaces = mark_special_token(s)
            try:
                if mode in [0, 2]:
                    sentence = self.tokenization_repair.correct(sentence)
                    sentence = reverse_tok_special(sentence)
                    s = self._restore_sentence(sentence, mark_replaces)                    
            except:
                pass
            
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
            # if mode in [0, 2]:
            #     sentence = self.tokenization_repair.correct(' '.join(tokens))
            # else:
            #     sentence = ' '.join(tokens)
            # result.append([sentence, data])
                # s = self.tokenization_repair.correct(s)
            result.append([' '.join(tokens), data])
        return result

    def _process_detection(self, words, detection_output):
        assert len(words) == len(detection_output)
        
        output = []
        for w, d in zip(words, detection_output):
            if w in string.punctuation:
                continue
            output.append(d)
        return output
          
    def _concat_spans(self, spans):
        if not spans:
            return spans
        sorted_spans = sorted(spans, key=lambda d: (d['start'], d["end"]))
        # print(sorted_spans)
        results = [sorted_spans[0]]
        for span in sorted_spans[1:]:
            if span["start"] <= results[-1]["end"]:
                results[-1]["end"] = span["end"]
            else:
                results.append(span)
        return results

    def _change_penalty(self, p_ins, p_del):
        self.p_del = p_del
        self.p_ins = p_ins
                
        self.tokenization_repair.insertion_penalty = -self.p_ins
        self.tokenization_repair.deletion_penalty = -self.p_del
        print("Change penalty")
        print(self.tokenization_repair.insertion_penalty)
        print(self.tokenization_repair.deletion_penalty)
            
    def _correct_sentences(self, input_sentences, mode):
        results = ""
        spans = []
        input_sentences = preprocess(input_sentences)        
        pairs = self._preprocess_sentences(input_sentences, mode)
        print(pairs)
        
        if mode == 0:
            repaired_sentences = ""
            for original_sentence, data in pairs:
                repaired_sentences += " " + self._match_case(original_sentence, original_sentence, data)
            repaired_sentences = repaired_sentences.strip()

            repaired_sentences = self.postprocessor.correct(repaired_sentences)
            tags, _, trgs = align_sentence(input_sentences, repaired_sentences)
            for tag, trg in zip(tags, trgs):
                if tag not in ["KEEP", "DELETE"]:
                    trg = re.sub(r'([\.\(\)\*\?\!\~\^\[\]\{\}])', r'\\\1', trg)
                    query = r'\b{}\b'.format(trg)
                    match = re.search(query, repaired_sentences)
                    if match:
                        start, end = match.span()
                        spans.append({
                            "start": start,
                            "end": end
                        })        

            spans = self._concat_spans(spans)            
            return repaired_sentences, spans
        else:
            for original_sentence, data in pairs:
                tmp = original_sentence.lower() 
                try:
                    words, _, _    = self._forward(tmp)
                    sentence = " ".join(words)
                except:
                    sentence = tmp  
                               
                results = results + ' ' + self._match_case(original_sentence, sentence, data)
                results = self.postprocessor.correct(results)

            results = results.strip()
            
            tags, _, trgs = align_sentence(input_sentences, results)
            tmp_idx = 0
            tmp_text = results
            for tag, trg in zip(tags, trgs):
                if tag not in ["KEEP", "DELETE"]:
                    trg = re.sub(r'([\.\(\)\*\?\!\~\^\[\]\{\}\,])', r'\\\1', trg)
                    query = r'{}'.format(trg)
                    match = re.search(query, tmp_text)
                    if match:
                        start, end = match.span()
                        spans.append({
                            "start": tmp_idx + start,
                            "end": tmp_idx + end
                        })
                        tmp_idx += end
                        tmp_text = tmp_text[end:]

            spans = self._concat_spans(spans)
            for span in spans:
                print(span)
                
            return results.strip(), spans        

    def _accum_span(self, spans, accum_size):
        for i in range(len(spans)):
            spans[i]["start"] += accum_size
            spans[i]["end"] += accum_size
        return spans   

    def correct(self, input_sentences, mode, p_ins=P_INS, p_del=P_DEL):
        if p_ins != self.p_ins or p_del != self.p_del:
            self._change_penalty(p_ins, p_del)
        return self._correct_sentences(input_sentences, mode)

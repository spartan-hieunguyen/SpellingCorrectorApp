import math

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from autocorrection.params import DEVICE, BERT_PRETRAINED
from transformers import AutoConfig, AutoModel

device = DEVICE

class PhoBertEncoder(nn.Module):
    def __init__(self, n_words: int, 
                n_labels_error: int,
                fine_tuned: bool = False, 
                use_detection_context: bool = True):
        super(PhoBertEncoder, self).__init__()
        self.bert_config = AutoConfig.from_pretrained(BERT_PRETRAINED, return_dict=True,
                                                         output_hidden_states=True)
        self.bert = AutoModel.from_pretrained(BERT_PRETRAINED, config=self.bert_config)
        self.d_hid = self.bert.config.hidden_size
        self.detection = nn.Linear(self.d_hid, n_labels_error)
        self.use_detection_context = use_detection_context
        if self.use_detection_context:
            self.detection_context_layer = nn.Sequential(
                nn.Softmax(dim=-1),
                nn.Linear(n_labels_error, self.d_hid)
            )
        self.max_n_subword = 30
        self.linear_subword_embedding = nn.Linear(self.max_n_subword * self.d_hid, self.d_hid)
        self.fine_tuned = fine_tuned
        self.correction = nn.Linear(self.d_hid, n_words)
        self.is_freeze_model()

    def is_freeze_model(self):
        for child in self.bert.children():
            for param in child.parameters():
                param.requires_grad = self.fine_tuned

    def merge_embedding(self, sequence_embedding: Tensor, sequence_split, mode='add'):
        sequence_embedding = sequence_embedding[1: sum(sequence_split) + 1]  # batch_size*seq_length*hidden_size
        embeddings = torch.split(sequence_embedding, sequence_split, dim=0)
        word_embeddings = pad_sequence(
            embeddings,
            padding_value=0,
            batch_first=True
        )
        if mode == 'avg':
            temp = torch.tensor(sequence_split).reshape(-1, 1).to(device)
            outputs = torch.div(torch.sum(word_embeddings, dim=1), temp)
        elif mode == 'add':
            outputs = torch.sum(word_embeddings, dim=1)
        elif mode == 'linear':
            embeddings = [
                torch.cat((
                    embedding_subword_tensor.reshape(-1),
                    torch.tensor([0] * (self.max_n_subword - embedding_subword_tensor.size(0)) * self.d_hid).to(device)
                ))
                for embedding_subword_tensor in embeddings
            ]
            embeddings = torch.stack(embeddings, dim=0)
            outputs = self.linear_subword_embedding(embeddings)
        else:
            raise Exception('Not Implemented')
        return outputs

    def forward(self, input_ids: Tensor,
                attention_mask: Tensor,
                batch_splits,
                token_type_ids: Tensor = None
                ):
        outputs = self.bert(input_ids, attention_mask)
        hidden_states = outputs.hidden_states
        stack_hidden_state = torch.stack(
                                    [hidden_states[-1], hidden_states[-2], hidden_states[-3], hidden_states[-4]],
                                    dim=0
                                )
        mean_hidden_state = torch.mean(stack_hidden_state, dim=0)
        outputs = pad_sequence(
            [self.merge_embedding(sequence_embedding, sequence_split) for sequence_embedding, sequence_split in
             zip(mean_hidden_state, batch_splits)],
            padding_value=0,
            batch_first=True
        )
        detection_outputs = self.detection(outputs)
        if self.use_detection_context:
            detection_context = self.detection_context_layer(detection_outputs)  # batch_size*seq_length*hidden_size
            outputs = outputs + detection_context

        correction_outputs = self.correction(outputs)
        return detection_outputs, correction_outputs

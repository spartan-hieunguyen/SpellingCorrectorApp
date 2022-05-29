import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from autocorrection.constants import DEVICE, BERT_PRETRAINED
from transformers import AutoConfig, AutoModel

device = DEVICE

# During training, we need a subsequent word mask that will prevent model to look into the future words when making predictions.
def generate_square_mask(sequence_size: int):
    mask = (torch.triu(torch.ones((sequence_size, sequence_size), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def generate_source_mask(src: Tensor, mask_token_id: int):
    src_mask = (src == mask_token_id)
    return src_mask


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


class GRUDetection(nn.Module):
    def __init__(self, n_words: int, 
                n_labels_error: int, 
                d_model: int = 512, 
                d_hid: int = 512, 
                n_layers: int = 2,
                bidirectional: bool = True, 
                dropout: float = 0.2):
        super(GRUDetection, self).__init__()
        self.word_embedding = nn.Embedding(n_words, d_model)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_hid,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.output_dim = d_hid * 2 if bidirectional else d_hid
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.output_dim, n_labels_error)

    def forward(self, src):
        """
        :param src: word error token ids
        :return: probability for each error type [batch_size, n_words, n_errors] and word error embedding [batch_size * seq_len * d_model]
        """
        embeddings = self.word_embedding(src)
        outputs, _ = self.gru(embeddings)  # batch_size*seq_length*(2*hidden_size)
        outputs = self.dropout(self.linear(outputs))
        return self.softmax(outputs), embeddings


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 256, 
                dropout: float = 0.1, 
                max_len: int = 400):

        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(dim=1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(100000) / d_model))
        self.position_encoding = torch.zeros(max_len, d_model).to(device)
        self.position_encoding[:, 0::2] = torch.sin(position * div_term)
        self.position_encoding[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x: Tensor) -> Tensor:
        """x: shape [batch_size, seq_length, embedding_dim] --> return [batch_size, seq_length, embedding_dim]"""
        x += self.position_encoding[:x.size(1)]
        return self.dropout(x)


class MaskedSoftBert(nn.Module):
    def __init__(self, n_words: int, 
                n_labels_error: int, 
                mask_token_id: int,
                n_head: int = 8, 
                n_layer_attn: int = 6, 
                d_model: int = 512, 
                d_hid: int = 512,
                n_layers_gru: int = 2, 
                bidirectional: bool = True, 
                dropout: float = 0.2):

        super(MaskedSoftBert, self).__init__()
        self.detection = GRUDetection(n_words=n_words,
                                      n_labels_error=n_labels_error,
                                      d_model=d_model,
                                      n_layers=n_layers_gru,
                                      bidirectional=bidirectional
                                      )
        self.position_encoding = PositionalEncoding(d_model, dropout, max_len=128)
        self.encoder_layer = nn.TransformerEncoderLayer(d_hid, n_head, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, n_layer_attn)
        self.mask_token_id = mask_token_id
        self.correction = nn.Linear(d_hid, n_words)

    def forward(self, src: Tensor,
                src_mask: Tensor = None,
                src_key_padding_mask: Tensor = None
                ):
        mask_embedding = self.detection.word_embedding(torch.tensor([[self.mask_token_id]]).to(device))
        detection_outputs, embeddings = self.detection(src)
        prob_correct_word = detection_outputs[:, :, 0].unsqueeze(2)  # batch_size * n_words *1
        # embedding: batch_size * n_words * d_model
        soft_mask_embedding = prob_correct_word * embeddings + (1 - prob_correct_word) * mask_embedding
        soft_mask_embedding = self.position_encoding(soft_mask_embedding)
        if src_mask is None or src_mask.size(0) != src.size(1):
            src_mask = generate_square_mask(src.size(1))

        if src_key_padding_mask is None:
            src_key_padding_mask = generate_source_mask(src, self.mask_token_id)
        outputs = self.transformer_encoder(
            soft_mask_embedding.transpose(0, 1),  # seq_len * batch_size * hidden_size
            mask=src_mask,  # seq_len * seq_len
            src_key_padding_mask=src_key_padding_mask  # batch_size*seq_len
        ).transpose(0, 1)  # batch_size * n_words * d_hid
        outputs += embeddings
        correction_outputs = self.correction(outputs)
        return detection_outputs, correction_outputs
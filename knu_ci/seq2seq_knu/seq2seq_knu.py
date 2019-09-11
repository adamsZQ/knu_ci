#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 6/9/19 8:10 PM
# @Author  : zchai
from typing import Dict, overload, List, Tuple

import torch
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.models import SimpleSeq2Seq
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, Attention
from allennlp.training.metrics import SequenceAccuracy, CategoricalAccuracy
from allennlp.nn import util

from torch.nn import GRUCell, Linear


class Seq2SeqKnu(SimpleSeq2Seq):

    def __init__(self, vocab: Vocabulary, source_embedder: TextFieldEmbedder, encoder: Seq2SeqEncoder, target_namespace,
                 decoder, attention, max_decoding_steps: int, cuda_device: int):
        super().__init__(vocab, source_embedder, encoder, max_decoding_steps, use_bleu=False)

        self._decoder = decoder

        self._attention = attention

        self.acc = CategoricalAccuracy()

        self.cuda_device = cuda_device

        self._target_namespace = target_namespace

        num_classes = self.vocab.get_vocab_size(self._target_namespace)

        self._output_projection_layer = Linear(self._decoder_output_dim * 3, num_classes)

    def forward(self,
                source_tokens: Dict[str, torch.LongTensor],
                gold_mentions,
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:

        state = self._encode(source_tokens)

        output_dict = self._forward_loop(state, gold_mentions, target_tokens)

        if not self.training:
            if target_tokens:
                logits = output_dict['logits']
                mention_mask = output_dict['mention_mask']
                target = target_tokens['tokens']

                self.acc(logits, target, mention_mask)

        return output_dict

    def _forward_loop(self,
                      state: Dict[str, torch.Tensor],
                      gold_mentions: torch.LongTensor,
                      target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        encoder_outputs = state['encoder_outputs']

        batch_size = source_mask.size()[0]

        max_input_sequence_length = source_mask.size()[1]
        # 下面两步将gold_mention用0扩充到 (batch_size, max_input_sequence_length)
        gold_mentions_expanded = torch.zeros(batch_size, max_input_sequence_length).cuda(self.cuda_device)
        gold_mentions_expanded[:, :gold_mentions.size()[1]] = gold_mentions

        mention_mask = util.get_text_field_mask({'gold_mentions': gold_mentions_expanded})

        for b in range(batch_size):
            encoder_output = encoder_outputs[b]
            gold_mention = gold_mentions_expanded[b]
            # 选择特定index的output，剩余的用0位置的output填充
            encoder_selected = torch.index_select(encoder_output, 0, gold_mention.long())

            if b == 0:
                encoder_resorted = encoder_selected.unsqueeze(0)
            else:
                encoder_resorted = torch.cat((encoder_resorted, encoder_selected.unsqueeze(0)), 0)

        decoder_outputs = self._decode(encoder_resorted, mention_mask)

        # 按照token一个个计算
        token_logits = []
        token_predictions = []
        for i in range(max_input_sequence_length):
            encoder_slice = encoder_resorted[:, i, :]

            decoder_hidden = decoder_outputs[:, i, :]

            # source_mask_slice = source_mask[:, i].float()

            encoder_weights = self._attention(decoder_hidden, encoder_outputs, source_mask.float())

            attended_output = util.weighted_sum(encoder_outputs, encoder_weights)

            hidden_attention_cat = torch.cat((decoder_hidden, attended_output, encoder_slice), -1)

            score = self._output_projection_layer(hidden_attention_cat)

            token_logits.append(score.unsqueeze(1))

            class_probabilities = F.softmax(score, dim=-1)

            # shape (predicted_classes): (batch_size,)
            _, predicted_classes = torch.max(class_probabilities, 1)

            last_predictions = predicted_classes

            token_predictions.append(last_predictions.unsqueeze(1))

        predictions = torch.cat(token_predictions, 1)
        # 裁切超过target长度的
        output_dict = {'predictions': predictions}

        if target_tokens:

            targets = target_tokens['tokens']
            target_length = targets.size()[1]

            predictions_slice = predictions[:, :target_length]
            output_dict = {'predictions': predictions_slice}

            target_length = targets.size()[1]
            logits = torch.cat(token_logits, 1)
            # 裁切超过target长度的
            logits_slice = logits[:, :target_length, :].contiguous()
            targets = targets.contiguous()
            mention_mask = mention_mask[:, :target_length].contiguous()
            loss = util.sequence_cross_entropy_with_logits(logits_slice.float(), targets, mention_mask.float())
            output_dict['loss'] = loss
            output_dict['logits'] = logits_slice
            output_dict['mention_mask'] = mention_mask

        return output_dict

    def _decode(self, encoder_output, decode_mask):
        decoder_outputs = self._decoder(encoder_output, decode_mask)
        return decoder_outputs

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if not self.training:
            all_metrics.update({'accuracy': self.acc.get_metric(reset=reset)})
        return all_metrics

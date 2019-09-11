#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/9/19 7:41 PM
# @Author  : zchai
from typing import Dict, List

import torch
from allennlp.models import SimpleSeq2Seq
from allennlp.training.metrics import CategoricalAccuracy, SequenceAccuracy, BooleanAccuracy
from allennlp.nn import util


class MySeq2Seq(SimpleSeq2Seq):
    """
    存在问题，要做序列标注需要改一点代码
    """
    
    def __init__(self, **kwargs):
        super(MySeq2Seq, self).__init__(**kwargs)

        self.acc = BooleanAccuracy()

    def forward(self,  # type: ignore
                source_tokens: Dict[str, torch.LongTensor],
                target_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:

        state = self._encode(source_tokens)

        if target_tokens:
            state = self._init_decoder_state(state)
            # The `_forward_loop` decodes the input sequence and computes the loss during training
            # and validation.
            output_dict = self._forward_loop(state, target_tokens)
        else:
            output_dict = {}

        if not self.training:
            state = self._init_decoder_state(state)
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)
            if target_tokens:
                # shape: (batch_size, beam_size, max_sequence_length)
                top_k_predictions = output_dict["predictions"]
                # shape: (batch_size, max_predicted_sequence_length)
                best_predictions = top_k_predictions[:, 0, :]
                # TODO decoder resulting different dimension matrix
                target_mask = util.get_text_field_mask(target_tokens)
                self.acc(best_predictions, target_tokens["tokens"], target_mask)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        all_metrics: Dict[str, float] = {}
        if not self.training:
            all_metrics.update(self.acc.get_metric(reset=reset))
        return all_metrics


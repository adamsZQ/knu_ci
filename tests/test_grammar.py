#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/9/19 4:19 PM
# @Author  : zchai
import json

import numpy as np
import torch
from allennlp.data import Instance
from allennlp.data.fields import ArrayField
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from knu_ci.seq2seq_knu.seq2seq_knu import Seq2SeqKnu


def test_grammar():
    label = torch.Tensor([[-1], [5], [7], [2], [9]]).float()
    '''
    tensor([[1.],
            [5.],
            [7.],
            [2.],
            [9.]])
    '''
    a = label[0][0]
    b = torch.Tensor([-1]).float()[0]
    boola = torch.equal(b, a)
    print('aaa')


def test_padding():
    a = [[1, 2, 3, 4], [1]]
    lengths = [len(e) for e in a]
    """pack"""
    x_emb_p = pad_packed_sequence(a, padding_value=-1, total_length=6, batch_first=True)
    """unpack: out"""
    out = pack_padded_sequence(x_emb_p, lengths, batch_first=True)  # (sequence, lengths)
    out = out[0]  #
    """unsort"""
    print(out)
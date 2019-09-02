#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/9/19 4:32 PM
# @Author  : zchai
import random

import torch
from torch import nn


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim,\
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers,\
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):

        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input_ = trg[0, :]

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input_, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input_ = (trg[t] if teacher_force else top1)

        return outputs

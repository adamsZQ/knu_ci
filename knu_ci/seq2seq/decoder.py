#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/9/19 4:19 PM
# @Author  : zchai
from torch import nn


class Decoder(nn.Module):

    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_, hidden, cell):

        input_ = input_.unsqueeze(0)

        embedded = self.dropout(self.embedding(input_))

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        pred = self.out(output.squeeze(0))

        return pred, hidden, cell

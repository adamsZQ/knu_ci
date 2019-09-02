#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/9/19 5:00 PM
# @Author  : zchai
import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import spacy

import random
import math
import time

# 设定torch的随机数种子，保证每次结果一样
from knu_ci.seq2seq.decoder import Decoder
from knu_ci.seq2seq.encoder import Encoder
from knu_ci.seq2seq.seq2seq import Seq2Seq
from knu_ci.utils import conf, logger

SEED = 1234

random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings (tokens) and reverses it
    """
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


def data_loader(batch_size):
    src_tokens = Field(tokenize=tokenize_de,
                       init_token='<sos>',
                       eos_token='<eos>',
                       lower=True)

    trg_tokens = Field(tokenize=tokenize_en,
                       init_token='<sos>',
                       eos_token='<eos>',
                       lower=True)

    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                        fields=(src_tokens, trg_tokens))

    src_tokens.build_vocab(train_data, min_freq=2)
    trg_tokens.build_vocab(train_data, min_freq=2)

    batch_size = batch_size

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=batch_size,
        device=device)

    return src_tokens, trg_tokens, train_iterator, valid_iterator, test_iterator


def init_weights(m):
    """
    初始化模型的所有参数
    :param m: model
    :return:
    """
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)  # turn off teacher forcing

            # trg = [trg sent len, batch size]
            # output = [trg sent len, batch size, output dim]

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            # trg = [(trg sent len - 1) * batch size]
            # output = [(trg sent len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def seq2seq_train():
    config = conf['train']
    batch_size = config['batch_size']
    enc_emb_dim = config['enc_emb_dim']
    dec_emb_dim = config['dec_emb_dim']
    hid_dim = config['hid_dim']
    enc_dropout = config['enc_dropout']
    dec_dropout = config['dec_dropout']
    n_layers = config['n_layers']

    src_tokens, trg_tokens, train_iterator, valid_iterator, test_iterator = data_loader(batch_size)

    input_dim = len(src_tokens.vocab)
    output_dim = len(trg_tokens.vocab)

    encoder = Encoder(input_dim, enc_emb_dim, hid_dim, n_layers, enc_dropout)
    decoder = Decoder(output_dim, dec_emb_dim, hid_dim, n_layers, dec_dropout)

    model = Seq2Seq(encoder, decoder, device).to(device)

    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters())

    pad_idx = trg_tokens.vocab.stoi['<pad>']

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    n_epochs = 10
    clip = 1

    best_valid_loss = float('inf')

    for epoch in range(n_epochs):

        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, clip)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut1-model.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


if __name__ == '__main__':
    logger.info('train.py test')
    seq2seq_train()



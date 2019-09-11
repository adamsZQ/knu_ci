#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/9/19 7:49 PM
# @Author  : zchai
import itertools
import os

import torch
from allennlp.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.activations import Activation
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from allennlp.modules.attention import LinearAttention
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.predictors import SimpleSeq2SeqPredictor
from allennlp.training.trainer import Trainer
from torch import optim


from knu_ci.my_logger import Logger
from knu_ci.seq2seq_allen.my_seq2seq import MySeq2Seq
from knu_ci.utils import conf, BASE_DIR


logger = Logger(__name__).get_logger()


class Seq2SeqAllen:

    def __init__(self):
        config = conf['seq2seq_allen']
        prefix = config['processed_data_prefix']
        train_file = config['train_data']
        valid_file = config['valid_data']
        src_embedding_dim = config['src_embedding_dim']
        trg_embedding_dim = config['trg_embedding_dim']
        hidden_dim = config['hidden_dim']

        if torch.cuda.is_available():
            cuda_device = 0
        else:
            cuda_device = -1

        self.reader = Seq2SeqDatasetReader(
                        source_tokenizer=WordTokenizer(),
                        target_tokenizer=WordTokenizer(),
                        source_token_indexers={'tokens': SingleIdTokenIndexer()},
                        target_token_indexers={'tokens': SingleIdTokenIndexer(namespace='target_tokens')})

        self.train_dataset = self.reader.read(os.path.join(prefix, train_file))
        self.valid_dataset = self.reader.read(os.path.join(prefix, valid_file))

        vocab = Vocabulary.from_instances(self.train_dataset + self.valid_dataset,
                                          min_count={'tokens': 3, 'target_tokens': 3})

        src_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                  embedding_dim=src_embedding_dim)

        encoder = PytorchSeq2SeqWrapper(
            torch.nn.LSTM(src_embedding_dim, hidden_dim, batch_first=True))

        source_embedder = BasicTextFieldEmbedder({"tokens": src_embedding})

        attention = LinearAttention(hidden_dim, hidden_dim, activation=Activation.by_name('tanh')())

        self.model = SimpleSeq2Seq(vocab=vocab, source_embedder=source_embedder, encoder=encoder,
                               max_decoding_steps=20,
                               target_embedding_dim=trg_embedding_dim,
                               target_namespace='target_tokens',
                               attention=attention,  # pass attention
                               use_bleu=True)

        optimizer = optim.Adam(self.model.parameters())
        iterator = BucketIterator(batch_size=32, sorting_keys=[("source_tokens", "num_tokens")])
        # 迭代器需要接受vocab，在训练时可以用vocab来index数据
        iterator.index_with(vocab)

        self.model.cuda(cuda_device)

        self.trainer = Trainer(model=self.model,
                               optimizer=optimizer,
                               iterator=iterator,
                               patience=10,
                               validation_metric="+accuracy",
                               train_dataset=self.train_dataset,
                               validation_dataset=self.valid_dataset,
                               num_epochs=1,
                               cuda_device=cuda_device)

    def train(self, epoch):

        for i in range(epoch):
            logger.info('Epoch: {}'.format(i))
            self.trainer.train()

            # predictor = SimpleSeq2SeqPredictor(self.model, self.reader)
            #
            # for instance in itertools.islice(self.valid_dataset, 10):
            #     logger.info('SOURCE:{}'.format(instance.fields['source_tokens'].tokens))
            #     logger.info('GOLD:{}'.format(instance.fields['target_tokens'].tokens))
            #     logger.info('PRED:{}'.format(predictor.predict_instance(instance)['predicted_tokens']))





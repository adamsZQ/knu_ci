#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/9/19 4:45 PM
# @Author  : zchai
import os

import torch
from allennlp.data import Vocabulary
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.modules import Embedding
from allennlp.modules.attention import LinearAttention
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import Activation
from allennlp.training import Trainer
from torch import optim
from torch.nn import GRUCell

from knu_ci.my_logger import Logger
from knu_ci.seq2seq_knu.my_seq_data_reader import MySeqDatasetReader
from knu_ci.seq2seq_knu.my_seq_predictor import MySeqPredictor
from knu_ci.seq2seq_knu.seq2seq_knu import Seq2SeqKnu
from knu_ci.utils import conf

logger = Logger(__name__).get_logger()


class KnuTrainer:

    def __init__(self, training=False):
        self.training = training
        config = conf['seq2seq_allen']
        prefix = config['processed_data_prefix']
        train_file = config['train_data']
        valid_file = config['valid_data']
        src_embedding_dim = config['src_embedding_dim']
        hidden_dim = config['hidden_dim']
        batch_size = config['batch_size']
        epoch = config['epoch']
        self.model_path = config['model']

        if torch.cuda.is_available():
            cuda_device = 0
        else:
            cuda_device = -1

        # 定义数据读取器，WordTokenizer代表按照空格分割，target的namespace用于生成输出层的vocab时不和source混在一起
        self.reader = MySeqDatasetReader(
                        source_tokenizer=WordTokenizer(),
                        target_tokenizer=WordTokenizer(),
                        source_token_indexers={'tokens': SingleIdTokenIndexer()},
                        target_token_indexers={'tokens': SingleIdTokenIndexer(namespace='target_tokens')})

        if training and self.model_path is not None:
            # 从文件中读取数据
            self.train_dataset = self.reader.read(os.path.join(prefix, train_file))
            self.valid_dataset = self.reader.read(os.path.join(prefix, valid_file))

            # 定义词汇
            self.vocab = Vocabulary.from_instances(self.train_dataset + self.valid_dataset,
                                                   min_count={'tokens': 3, 'target_tokens': 3})
        elif not training:
            self.vocab = Vocabulary.from_files(self.model_path)

        # 定义embedding层
        src_embedding = Embedding(num_embeddings=self.vocab.get_vocab_size('tokens'),
                                  embedding_dim=src_embedding_dim)

        # 定义encoder，这里使用的是BiGRU
        encoder = PytorchSeq2SeqWrapper(
            torch.nn.GRU(src_embedding_dim, hidden_dim // 2, batch_first=True, bidirectional=True))

        # 定义decoder，这里使用的是GRU，因为decoder的输入需要和encoder的输出一致
        decoder = PytorchSeq2SeqWrapper(torch.nn.GRU(hidden_dim, hidden_dim, batch_first=True))
        # 将index 映射到 embedding上，tokens与data reader中用的TokenInder一致
        source_embedder = BasicTextFieldEmbedder({"tokens": src_embedding})

        # 线性Attention层
        attention = LinearAttention(hidden_dim, hidden_dim, activation=Activation.by_name('tanh')())

        # 定义模型
        self.model = Seq2SeqKnu(vocab=self.vocab, source_embedder=source_embedder, encoder=encoder, target_namespace = 'target_tokens',
                                decoder=decoder, attention=attention, max_decoding_steps=20, cuda_device=cuda_device)

        # 判断是否训练
        if training and self.model_path is not None:
            optimizer = optim.Adam(self.model.parameters())
            # sorting_keys代表batch的时候依据什么排序
            iterator = BucketIterator(batch_size=batch_size, sorting_keys=[("source_tokens", "num_tokens")])
            # 迭代器需要接受vocab，在训练时可以用vocab来index数据
            iterator.index_with(self.vocab)

            self.model.cuda(cuda_device)

            # 定义训练器
            self.trainer = Trainer(model=self.model,
                                   optimizer=optimizer,
                                   iterator=iterator,
                                   patience=10,
                                   validation_metric="+accuracy",
                                   train_dataset=self.train_dataset,
                                   validation_dataset=self.valid_dataset,
                                   serialization_dir=self.model_path,
                                   num_epochs=epoch,
                                   cuda_device=cuda_device)
        elif not training:
            with open(os.path.join(self.model_path, 'best.th'), 'rb') as f:
                self.model.load_state_dict(torch.load(f))
            self.model.cuda(cuda_device)
            self.predictor = MySeqPredictor(self.model, dataset_reader=self.reader)

    def train(self):
        if self.training:
            self.trainer.train()
            self.vocab.save_to_files(self.model_path)
        else:
            logger.warning('Model is not in training state!')

    def predict(self, sentence):
        if not self.training:
            return self.predictor.predict_json(sentence)
        else:
            logger.warning('Mode is in training state!')



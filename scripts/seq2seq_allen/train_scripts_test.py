#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/9/19 10:55 AM
# @Author  : zchai
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
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.activations import Activation
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from allennlp.modules.attention import LinearAttention, BilinearAttention, DotProductAttention
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, StackedSelfAttentionEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.predictors import SimpleSeq2SeqPredictor
from allennlp.training.trainer import Trainer
from torch import optim

from knu_ci.my_logger import Logger
from knu_ci.utils import conf, BASE_DIR


logger = Logger(__name__).get_logger()

config = conf['seq2seq_allen']
train_file = config['train_data']
valid_file = config['valid_data']
src_embedding_dim = config['src_embedding_dim']
trg_embedding_dim = config['trg_embedding_dim']
hidden_dim = config['hidden_dim']

cuda_device = 1 if torch.cuda.is_available() else 0

reader = Seq2SeqDatasetReader(
                source_tokenizer=WordTokenizer(),
                target_tokenizer=WordTokenizer(),
                source_token_indexers={'tokens': SingleIdTokenIndexer()},
                target_token_indexers={'tokens': SingleIdTokenIndexer(namespace='target_tokens')})

train_dataset = reader.read(os.path.join(BASE_DIR, train_file))
valid_dataset = reader.read(os.path.join(BASE_DIR, valid_file))

vocab = Vocabulary.from_instances(train_dataset + valid_dataset,
                                  min_count={'tokens': 3, 'target_tokens': 3})

src_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                          embedding_dim=src_embedding_dim)

encoder = PytorchSeq2SeqWrapper(
    torch.nn.LSTM(src_embedding_dim, hidden_dim, batch_first=True))

source_embedder = BasicTextFieldEmbedder({"tokens": src_embedding})

attention = LinearAttention(hidden_dim, hidden_dim, activation=Activation.by_name('tanh')())

model = SimpleSeq2Seq(vocab, source_embedder, encoder, max_decoding_steps=20,
                           target_embedding_dim=trg_embedding_dim,
                           target_namespace='target_tokens',
                           attention=attention,  # pass attention
                           beam_size=8,
                           use_bleu=True)

optimizer = optim.Adam(model.parameters())
iterator = BucketIterator(batch_size=32, sorting_keys=[("source_tokens", "num_tokens")])
iterator.index_with(vocab)

trainer = Trainer(model=model,
                       optimizer=optimizer,
                       iterator=iterator,
                       patience=10,
                       train_dataset=train_dataset,
                       validation_dataset=valid_dataset,
                       num_epochs=1,
                       cuda_device=cuda_device)

for i in range(10):
    logger.info('Epoch: {}'.format(i))
    trainer.train()

    predictor = SimpleSeq2SeqPredictor(model, reader)

    for instance in itertools.islice(valid_dataset, 10):
        logger.info('SOURCE:', instance.fields['source_tokens'].tokens)
        logger.info('GOLD:', instance.fields['target_tokens'].tokens)
        logger.info('PRED:', predictor.predict_instance(instance)['predicted_tokens'])





#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 7/9/19 2:20 PM
# @Author  : zchai
import csv
import json
from typing import Iterable, Dict

import numpy as np
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import DatasetReader, Instance, Tokenizer, TokenIndexer, Token
from allennlp.data.fields import TextField, ArrayField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer

from knu_ci.my_logger import Logger

logger = Logger(__name__).get_logger()


class MySeqDatasetReader(DatasetReader):

    def __init__(self,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token: bool = True,
                 delimiter: str = "\t",
                 lazy: bool = False):
        super().__init__(lazy)

        self._source_tokenizer = source_tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_add_start_token = source_add_start_token
        self._delimiter = delimiter

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, row in enumerate(csv.reader(data_file, delimiter=self._delimiter)):
                if len(row) != 3:
                    raise ConfigurationError("Invalid line format: %s (line number %d)" % (row, line_num + 1))
                source_sequence, gold_mention_sequence, target_sequence = row
                yield self.text_to_instance(source_sequence, gold_mention_sequence, target_sequence)

    def text_to_instance(self, source_string: str, gold_mention_string: str, target_string: str = None) -> Instance:
        tokenized_source = self._source_tokenizer.tokenize(source_string)
        if self._source_add_start_token:
            tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)

        if self._source_add_start_token:
            # 如果添加了start token就需要将mention的index + 1
            gold_mention_array = np.array([ind + 1 for ind in json.loads(gold_mention_string)])
        else:
            gold_mention_array = np.array(json.loads(gold_mention_string))
        gold_mention_field = ArrayField(gold_mention_array, padding_value=0)

        if target_string is not None and gold_mention_string is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            # 不加入起始和终止符号
            # tokenized_target.insert(0, Token(START_SYMBOL))
            # tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(tokenized_target, self._target_token_indexers)

            return Instance({"source_tokens": source_field, "gold_mentions": gold_mention_field,
                             "target_tokens": target_field})
        else:
            return Instance({'source_tokens': source_field, "gold_mentions": gold_mention_field})

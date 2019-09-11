#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/9/19 8:00 PM
# @Author  : zchai
from allennlp.common import JsonDict
from allennlp.data import Instance, DatasetReader
from allennlp.models import Model
from allennlp.predictors import Predictor


class MySeqPredictor(Predictor):

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._model = model
        self._dataset_reader = dataset_reader

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        source_tokens = json_dict['source_tokens']
        gold_mentions = json_dict['gold_mentions']

        return self._dataset_reader.text_to_instance(source_tokens, gold_mentions)
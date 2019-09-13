#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 12/9/19 10:58 AM
# @Author  : zchai
import csv
import os

from knu_ci.seq2seq_allen.seq2seq_allen import logger
from knu_ci.utils import conf


def analysis():
    config = conf['seq2seq_allen']
    data_file = os.path.join(config['processed_data_prefix'], config['train_data'])

    character_list = []
    with open(data_file, mode='r') as f:

        for line_num, row in enumerate(csv.reader(f, delimiter='\t')):
            source_sequence, gold_mention_sequence, target_sequence = row

            characters = target_sequence.split(' ')

            for character in characters:
                if character not in character_list:
                    character_list.append(character)

    logger.info(len(character_list))
    logger.info(character_list)


if __name__ == '__main__':
    analysis()
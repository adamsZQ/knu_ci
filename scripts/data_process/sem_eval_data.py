#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 5/9/19 10:53 AM
# @Author  : zchai
import csv
import json
import os

from knu_ci.utils import conf


def process():
    config = conf['seq2seq_allen']
    origin_prefix = config['origin_data_prefix']
    target_prefix = config['processed_data_prefix']
    train_file = config['train_data']
    valid_file = config['valid_data']
    test_file = config['test_data']

    data_list = []
    for file in [train_file, valid_file, test_file]:
        with open(os.path.join(origin_prefix, file), mode='r', encoding='utf-8') as f:
            data_json = json.loads(f.read())

        for episode in data_json['episodes']:
            for scene in episode['scenes']:
                for utterance in scene['utterances']:
                    for token, character_entity in zip(utterance['tokens'], utterance['character_entities']):
                        sentence = ' '.join(token)
                        chara_ent = []
                        chara_ind = []
                        for ce in character_entity:
                            chara_ent.append(ce[-1].replace(' ', ''))
                            chara_ind.append(int(ce[0]))
                        chara_ent = ' '.join(chara_ent)

                        assert len(chara_ent) == len(chara_ent)
                        if len(chara_ind) != 0:
                            data_list.append((sentence, chara_ind, chara_ent))

        portion = os.path.splitext(file)
        file = portion[0] + '.tsv'
        with open(os.path.join(target_prefix, file), mode='w', encoding='utf-8') as w:
            tsv_v = csv.writer(w, delimiter='\t')
            for sentence, chara_ind, chara_ent in data_list[:100]:
                tsv_v.writerow([sentence, chara_ind, chara_ent])


if __name__ == '__main__':
    process()
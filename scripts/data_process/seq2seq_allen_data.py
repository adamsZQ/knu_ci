#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/9/19 9:48 AM
# @Author  : zchai
import csv
import os

from knu_ci.utils import BASE_DIR


def data_process():
    train_de_rel_path = '.data/multi30k/train.de'
    train_en_rel_path = '.data/multi30k/train.en'

    val_de_rel_path = '.data/multi30k/val.de'
    val_en_rel_path = '.data/multi30k/val.en'

    with open(os.path.join(BASE_DIR, train_de_rel_path), mode='r', encoding='utf-8') as de:
        train_de_list = de.readlines()

    with open(os.path.join(BASE_DIR, train_en_rel_path), mode='r', encoding='utf-8') as en:
        train_en_list = en.readlines()

    with open(os.path.join(BASE_DIR, val_de_rel_path), mode='r', encoding='utf-8') as de_val:
        val_de_list = de_val.readlines()

    with open(os.path.join(BASE_DIR, val_en_rel_path), mode='r', encoding='utf-8') as en_val:
        val_en_list = en_val.readlines()

    with open(os.path.join(BASE_DIR, "data/train.tsv"), mode='w', encoding='utf-8') as t:
        tsv_t = csv.writer(t, delimiter='\t')
        tsv_t.writerow(['de', 'en'])
        for train_de, train_en in zip(train_de_list, train_en_list):
            tsv_t.writerow([train_de, train_en])

    with open(os.path.join(BASE_DIR, "data/valid.tsv"), mode='w', encoding='utf-8') as v:
        tsv_v = csv.writer(v, delimiter='\t')
        tsv_v.writerow(['de', 'en'])
        for val_de, val_en in zip(val_de_list, val_en_list):
            tsv_v.writerow([val_de, val_en])


if __name__ == '__main__':
    data_process()
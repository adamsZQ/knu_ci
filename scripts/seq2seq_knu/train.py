#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/9/19 4:50 PM
# @Author  : zchai
import sys
import os

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CUR_PATH, '../../')))

from knu_ci.seq2seq_knu.trainer import KnuTrainer


def train():
    model = KnuTrainer(training=True)
    model.train()


if __name__ == '__main__':
    train()
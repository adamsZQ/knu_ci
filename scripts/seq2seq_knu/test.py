#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/9/19 7:22 PM
# @Author  : zchai
import sys
import os

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(CUR_PATH, '../../')))

from knu_ci.seq2seq_knu.trainer import KnuTrainer, logger


def evaluate():
    model = KnuTrainer(training=False)

    final_metrics = model.evaluate()

    logger.info(final_metrics)


if __name__ == '__main__':
    evaluate()
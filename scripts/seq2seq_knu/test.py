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
    result = model.predict({'source_tokens': "All right Joey , be nice .",
                            'gold_mentions': '[2]'})
    # JoeyTribbiani
    logger.info(result)

    result = model.predict({'source_tokens': "This guy says hello , I wan na kill myself",
                            'gold_mentions': '[1, 5, 9]'})
    #  RossGeller JoeyTribbiani JoeyTribbiani
    logger.info(result)

    result = model.predict({'source_tokens': "Let me get you some coffee .",
                            'gold_mentions': '[1, 3]'})
    # MonicaGeller RossGeller
    logger.info(result)


if __name__ == '__main__':
    evaluate()
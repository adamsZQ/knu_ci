#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/9/19 7:22 PM
# @Author  : zchai
from knu_ci.seq2seq_knu.trainer import KnuTrainer, logger


def evaluate():
    model = KnuTrainer(training=False)
    result = model.predict({'source_tokens': "There 's got ta be something wrong with him !",
                            'gold_mentions': '[8]'})
    logger.info(result)


if __name__ == '__main__':
    evaluate()
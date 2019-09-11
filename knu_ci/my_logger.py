#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/9/19 3:16 PM
# @Author  : zchai
import logging
import os

from knu_ci.utils import conf, BASE_DIR


class Logger(object):

    def __init__(self, logger_name):

        log_level = conf['log']['level']
        log_rel_file = conf['log']['file_name']
        prefix = os.path.join(BASE_DIR, 'log')
        if not os.path.exists(prefix):
            os.mkdir(prefix)
        log_file = os.path.join(prefix, log_rel_file)

        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(log_level)

        # 创建一个handler用于写入日志文件
        file_handler = logging.FileHandler(log_file)

        # 创建一个handler用于输出控制台
        console_handler = logging.StreamHandler()

        # 定义handler的输出格式
        formatter = logging.Formatter(
            '[%(asctime)s] [%(filename)s file line:%(lineno)d] - %(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 给logger添加handler
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger


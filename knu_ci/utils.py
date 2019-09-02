#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-08-23 09:58
# @Author  : zchai

import yaml
import logging
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_config(config_file):
    """
    解析yaml配置文件
    Args:
        config_file:yaml配置文件路径
    Returns:
    解析后的配置对象
    """
    with open(config_file, 'r') as config:
        config = yaml.load(config)
    return config


def _get_logger(config):
    log_level = logging.DEBUG if config['log']['level'] == 'DEBUG' else logging.INFO

    log_file = os.path.join(BASE_DIR, 'log/central_controller.log')
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))

    logging.basicConfig(filename=log_file,
                        filemode='a',
                        format='%(asctime)s %(name)s %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=log_level)

    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.INFO)

    return _logger


conf_file = os.path.join(BASE_DIR, 'conf/config.yaml')
conf = parse_config(conf_file)
logger = _get_logger(conf)


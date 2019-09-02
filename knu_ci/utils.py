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


conf_file = os.path.join(BASE_DIR, 'conf/config.yaml')
conf = parse_config(conf_file)


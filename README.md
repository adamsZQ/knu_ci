# knu_ci
主要为复现Knu_ci这篇论文，以及相关的前置seq2seq和allen nlp的学习
论文连接如下：
https://www.aclweb.org/anthology/S18-1107

## 代码结构
.
|-- ./conf
|   `-- ./conf/config.yaml  配置文件
|-- ./knu_ci
|   |-- ./knu_ci/__init__.py
|   |-- ./knu_ci/my_logger.py  logger文件
|   |-- ./knu_ci/seq2seq  torch写（抄）的seq2seq
|   |-- ./knu_ci/seq2seq_allen  基于allen_nlp写（抄）的seq2seq
|   |-- ./knu_ci/seq2seq_knu   基于allen_nlp复现的knu ci的代码
|   `-- ./knu_ci/utils.py
|-- ./scripts
|   |-- ./scripts/__init__.py
|   |-- ./scripts/data_process  数据处理相关
|   |-- ./scripts/seq2seq    seq2seq训练脚本
|   |-- ./scripts/seq2seq_allen   训练脚本
|   `-- ./scripts/seq2seq_knu   训练脚本
`-- ./tests
    |-- ./tests/__init__.py
    |-- ./tests/test_grammar.py   语法测试文件
    `-- ./tests/test_log.py   log测试文件

## 数据

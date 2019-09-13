# knu_ci
主要为复现Knu_ci这篇论文，以及相关的前置seq2seq和allen nlp的学习
论文连接如下：
https://www.aclweb.org/anthology/S18-1107

## 代码结构
```
.
|-- conf
|   `-- config.yaml
|-- knu_ci
|   |-- __init__.py
|   |-- my_logger.py
|   |-- seq2seq   # by torch
|   |-- seq2seq_allen   # by allen nlp
|   |-- seq2seq_knu   # by allen nlp
|   `-- utils.py
|-- scripts
|   |-- __init__.py
|   |-- data_process
|   |-- seq2seq
|   |-- seq2seq_allen
|   `-- seq2seq_knu
`-- tests
    |-- __init__.py
    |-- test_grammar.py
    `-- test_log.py
```

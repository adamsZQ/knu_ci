# knu_ci
主要为复现Knu_ci这篇论文，以及相关的前置seq2seq和allen nlp的学习
论文连接如下：
https://www.aclweb.org/anthology/S18-1107

## 代码结构
.
|-- ./conf
|   `-- ./conf/config.yaml
|-- ./data
|   `-- ./data/de_en
|-- ./knu_ci
|   |-- ./knu_ci/__init__.py
|   |-- ./knu_ci/my_logger.py
|   |-- ./knu_ci/seq2seq
|   |-- ./knu_ci/seq2seq_allen
|   |-- ./knu_ci/seq2seq_knu
|   `-- ./knu_ci/utils.py
|-- ./log
|   `-- ./log/main.log
|-- ./scripts
|   |-- ./scripts/__init__.py
|   |-- ./scripts/data_process
|   |-- ./scripts/seq2seq
|   |-- ./scripts/seq2seq_allen
|   `-- ./scripts/seq2seq_knu
`-- ./tests
    |-- ./tests/__init__.py
    |-- ./tests/test_grammar.py
    `-- ./tests/test_log.py


## 数据

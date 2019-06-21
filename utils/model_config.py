# coding=utf-8
class Config(object):
    def __init__(self):
        self.num_train_groups = 140000  # 前9万组问题拿来做训练
        self.max_len = 32
        self.batch_size = 100
        self.min_count = 5
        self.word_size = 128
        self.epochs = 30  # amsoftmax需要25个epoch，其它需要20个epoch

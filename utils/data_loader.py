# coding=utf-8
import pandas as pd
from utils.tools import strQ2B, construct_dict
from utils.model_config import Config
import numpy as np


class DataLoader(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.config = Config()
        self._process_raw_data()

    def string2id(self, s):
        ids = [self.char2id.get(i, 1) for i in s[:self.config.max_len]]
        padded_ids = ids + [0] * (self.config.max_len - len(ids))
        return padded_ids

    def _process_raw_data(self):
        raw_data = pd.read_csv(self.data_path, encoding='utf-8', header=None)
        # 全角转半角
        raw_data[1] = raw_data[1].apply(strQ2B)
        # 大写转小写
        raw_data[1] = raw_data[1].str.lower()
        # 构建字2id的字典, 只保留min_count以上的字
        self.char2id, self.max_feature = construct_dict(raw_data[1], self.config.min_count)
        raw_data[2] = raw_data[1].apply(self.string2id)
        train_data = raw_data[raw_data[0] < self.config.num_train_groups]
        train_data = train_data.sample(frac=1)
        self.x_train = np.array(list(train_data[2]))
        self.y_train = np.array(list(train_data[0])).reshape((-1, 1))
        valid_data = raw_data[raw_data[0] >= self.config.num_train_groups]
        self.x_valid = np.array(list(valid_data[2]))
        self.y_valid = np.array(list(valid_data[0])).reshape((-1, 1))
        # id与组别之间的映射
        self.valid_id2g = dict(zip(valid_data.index - valid_data.index[0], valid_data[0]))

    def process_test_data(self, test_data_path):
        test_data = pd.read_csv(test_data_path, encoding='utf-8', header=None)
        test_data[1] = test_data[1].apply(strQ2B)
        test_data[1] = test_data[1].str.lower()
        test_data[2] = test_data[1].apply(self.string2id)
        x_test = np.array(list(test_data[2]))
        y_test = np.array(list(test_data[0])).reshape((-1, 1))
        # id与组别之间的映射
        test_id2g = dict(zip(test_data.index - test_data.index[0], test_data[0]))
        return test_data, x_test, y_test, test_id2g

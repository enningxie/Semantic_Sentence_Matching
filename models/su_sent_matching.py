# coding=utf-8
from utils.model_config import Config
from keras.layers import Input, Embedding, CuDNNGRU, LSTM, Lambda, Dense, Bidirectional
from keras.models import Model
import keras.backend as K
from keras.constraints import unit_norm


class SentMatching(object):
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.config = Config()
        self._get_train_model()

    def _get_train_model(self):
        # 正式模型，基于GRU的分类器
        x_in = Input(shape=(self.config.max_len,))
        x_embedded = Embedding(self.data_loader.max_feature + 2,
                               self.config.word_size)(x_in)
        x = Bidirectional(LSTM(64))(x_embedded)
        # x = CuDNNGRU(self.config.word_size)(x_embedded)
        x = Lambda(lambda x: K.l2_normalize(x, 1))(x)

        pred = Dense(self.config.num_train_groups,
                     use_bias=False,
                     kernel_constraint=unit_norm())(x)

        self.encoder = Model(x_in, x)  # 最终的目的是要得到一个编码器
        self.model = Model(x_in, pred)  # 用分类问题做训练

    def get_ranking_model(self, data_size):
        # 为验证集的排序准备
        # 实际上用numpy写也没有问题，但是用Keras写能借助GPU加速
        x_in = Input(shape=(self.config.word_size,))
        x = Dense(data_size, use_bias=False)(x_in)  # 计算相似度
        x = Lambda(lambda x: K.tf.nn.top_k(x, 11)[1])(x)  # 取出topk的下标
        ranking_model = Model(x_in, x)
        return ranking_model

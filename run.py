# coding=utf-8
import os
import numpy as np
from keras.callbacks import Callback
import json
from utils.data_loader import DataLoader
from models.su_sent_matching import SentMatching
from utils.model_config import Config
from models.margin_softmax import sparse_amsoftmax_loss
from utils.tools import strQ2B
from keras.models import Model
from sklearn.cluster import KMeans

ROOT_DIR = os.getcwd()
TRAIN_DATA_PATH = os.path.join(ROOT_DIR, 'data', 'LCQMC.csv')
TEST_DATA_PATH = os.path.join(ROOT_DIR, 'data', 'valid_data_02.csv')
MODEL_PATH = os.path.join(ROOT_DIR, 'cache_data', 'sent_sim_amsoftmax_bilstm.h5')
PATH_FOR_ENCODER = os.path.join(ROOT_DIR, 'data', 'round_1.csv')


def evaluate_test(ranking_model, test_vec, id2g):  # 评测函数
    print('testing...')
    ranking_model.set_weights([test_vec.T])  # 载入句向量为权重
    sorted_result = ranking_model.predict(test_vec,
                                          verbose=True,
                                          batch_size=1000)  # 计算topk
    new_result = np.vectorize(lambda s: id2g[s])(sorted_result)
    return cal_acc(new_result)


def cal_acc(tmp_result):
    top1_acc = top5_acc = top10_acc = 0.
    _ = tmp_result[:, 0] != tmp_result[:, 0]  # 生成一个全为False的向量
    for i in range(10):  # 注意按照相似度排序的话，第一个就是输入句子（全匹配）
        _ = _ + (tmp_result[:, 0] == tmp_result[:, i + 1])
        if i + 1 == 1:
            top1_acc = 1. * _.sum() / len(_)
        elif i + 1 == 5:
            top5_acc = 1. * _.sum() / len(_)
        elif i + 1 == 10:
            top10_acc = 1. * _.sum() / len(_)
    return top1_acc, top5_acc, top10_acc


def train(model, encoder, ranking_model, data_loader, config):
    model.compile(loss=sparse_amsoftmax_loss,
                  optimizer='adam',
                  metrics=['sparse_categorical_accuracy'])

    def evaluate():  # 评测函数
        print('validing...')
        valid_vec = encoder.predict(data_loader.x_valid,
                                    verbose=True,
                                    batch_size=1000)  # encoder计算句向量
        ranking_model.set_weights([valid_vec.T])  # 载入句向量为权重
        sorted_result = ranking_model.predict(valid_vec,
                                              verbose=True,
                                              batch_size=1000)  # 计算topk
        new_result = np.vectorize(lambda s: data_loader.valid_id2g[s])(sorted_result)
        return cal_acc(new_result)

    # 定义Callback器，计算验证集的acc，并保存最优模型
    class Evaluate(Callback):
        def __init__(self):
            self.accs = {'top1': [], 'top5': [], 'top10': []}
            self.highest = 0.

        def on_epoch_end(self, epoch, logs=None):
            top1_acc, top5_acc, top10_acc = evaluate()
            self.accs['top1'].append(top1_acc)
            self.accs['top5'].append(top5_acc)
            self.accs['top10'].append(top10_acc)
            if top1_acc >= self.highest:  # 保存最优模型权重
                self.highest = top1_acc
                model.save_weights(os.path.join(ROOT_DIR, 'cache_data', 'sent_sim_amsoftmax_test.h5'))
            json.dump({'accs': self.accs, 'highest_top1': self.highest},
                      open('valid_amsoftmax.log', 'w'), indent=4)
            print('top1_acc: %s, top5_acc: %s, top10_acc: %s' % (top1_acc, top5_acc, top10_acc))

    # train step
    history = model.fit(data_loader.x_train,
                        data_loader.y_train,
                        batch_size=config.batch_size,
                        epochs=config.epochs,
                        callbacks=[Evaluate()])
    return model, encoder


def evaluate_(encoder, test_data_path, data_loader, sent_models):
    # test step
    test_data, x_test, y_test, test_id2g = data_loader.process_test_data(test_data_path)
    ranking_model = sent_models.get_ranking_model(y_test.shape[0])
    test_vec = encoder.predict(x_test,
                               verbose=True,
                               batch_size=1000)  # encoder计算句向量
    top1_acc, top5_acc, top10_acc = evaluate_test(ranking_model, test_vec, test_id2g)
    print('top1_acc: %s, top5_acc: %s, top10_acc: %s' % (top1_acc, top5_acc, top10_acc))
    return test_data, test_vec


def predict(encoder, data_loader, test_data, test_vec, s):
    v = encoder.predict(np.array([data_loader.string2id(strQ2B(s).lower())]))[0]
    sims = np.dot(test_vec, v)
    for i in sims.argsort()[-10:][::-1]:
        print(test_data.iloc[i][1], sims[i])


def main(train_flag=False, test_flag=False, use_encoder=False):
    # before train
    config = Config()
    data_loader = DataLoader(TRAIN_DATA_PATH)
    data_loader.process_raw_data()
    sent_models = SentMatching(data_loader)
    model = sent_models.model
    encoder = sent_models.encoder

    if train_flag:
        ranking_model = sent_models.get_ranking_model(data_loader.y_valid.shape[0])
        model, encoder = train(model, encoder, ranking_model, data_loader, config)
    else:
        model.load_weights(MODEL_PATH)
        encoder = Model(inputs=model.input,
                        outputs=model.get_layer(index=3).output)

    if test_flag:
        # 测试文件样例大小最小：11
        test_data, test_vec = evaluate_(encoder, TEST_DATA_PATH, data_loader, sent_models)

        # test_data, x_test, y_test, test_id2g = data_loader.process_test_data(TEST_DATA_PATH)
        # test_vec = encoder.predict(x_test,
        #                            verbose=True,
        #                            batch_size=1000)  # encoder计算句向量

        while True:
            input_sent = input()
            predict(encoder, data_loader, test_data, test_vec, input_sent)

    # todo
    if use_encoder:
        tmp_data, x_tmp, y_tmp, tmp_id2g = data_loader.process_test_data(PATH_FOR_ENCODER)
        tmp_vec = encoder.predict(x_tmp,
                                  verbose=True,
                                  batch_size=1000)  # encoder计算句向量
        print(tmp_vec.shape)
        print(tmp_vec[0].shape)
        sims = np.dot(tmp_vec, tmp_vec[0])
        for i in sims.argsort()[-1:][::-1]:
            print(tmp_data.iloc[i][1], sims[i])

        y_pred = KMeans(n_clusters=3, random_state=42).fit_predict(tmp_vec)
        print(y_pred)

    
if __name__ == '__main__':
    main(use_encoder=True)

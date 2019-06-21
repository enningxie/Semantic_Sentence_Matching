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


ROOT_DIR = os.getcwd()


# # 定义Callback器，计算验证集的acc，并保存最优模型
# class Evaluate(Callback):
#     def __init__(self):
#         self.accs = {'top1': [], 'top5': [], 'top10': []}
#         self.highest = 0.
#
#     def on_epoch_end(self, epoch, logs=None):
#         top1_acc, top5_acc, top10_acc = evaluate()
#         self.accs['top1'].append(top1_acc)
#         self.accs['top5'].append(top5_acc)
#         self.accs['top10'].append(top10_acc)
#         if top1_acc >= self.highest:  # 保存最优模型权重
#             self.highest = top1_acc
#             model.save_weights(os.path.join(ROOT_DIR, 'cache_data', 'sent_sim_amsoftmax_test.h5'))
#         json.dump({'accs': self.accs, 'highest_top1': self.highest},
#                   open('valid_amsoftmax.log', 'w'), indent=4)
#         print('top1_acc: %s, top5_acc: %s, top10_acc: %s' % (top1_acc, top5_acc, top10_acc))
#
#
# def evaluate():  # 评测函数
#     print('validing...')
#     valid_vec = encoder.predict(x_valid,
#                                 verbose=True,
#                                 batch_size=1000)  # encoder计算句向量
#     ranking_model.set_weights([valid_vec.T])  # 载入句向量为权重
#     sorted_result = ranking_model.predict(valid_vec,
#                                           verbose=True,
#                                           batch_size=1000)  # 计算topk
#     new_result = np.vectorize(lambda s: valid_id2g[s])(sorted_result)
#     _ = new_result[:, 0] != new_result[:, 0]  # 生成一个全为False的向量
#     for i in range(10):  # 注意按照相似度排序的话，第一个就是输入句子（全匹配）
#         _ = _ + (new_result[:, 0] == new_result[:, i + 1])
#         if i + 1 == 1:
#             top1_acc = 1. * _.sum() / len(_)
#         elif i + 1 == 5:
#             top5_acc = 1. * _.sum() / len(_)
#         elif i + 1 == 10:
#             top10_acc = 1. * _.sum() / len(_)
#
#     return top1_acc, top5_acc, top10_acc

def evaluate_test(encoder, ranking_model, test_vec, id2g):  # 评测函数
    print('validing...')
    ranking_model.set_weights([test_vec.T])  # 载入句向量为权重
    sorted_result = ranking_model.predict(test_vec,
                                          verbose=True,
                                          batch_size=1000)  # 计算topk
    new_result = np.vectorize(lambda s: id2g[s])(sorted_result)
    _ = new_result[:, 0] != new_result[:, 0]  # 生成一个全为False的向量
    for i in range(10):  # 注意按照相似度排序的话，第一个就是输入句子（全匹配）
        _ = _ + (new_result[:, 0] == new_result[:, i + 1])
        if i + 1 == 1:
            top1_acc = 1. * _.sum() / len(_)
        elif i + 1 == 5:
            top5_acc = 1. * _.sum() / len(_)
        elif i + 1 == 10:
            top10_acc = 1. * _.sum() / len(_)

    return top1_acc, top5_acc, top10_acc


def main():
    # before train
    config = Config()
    train_data_path = os.path.join(ROOT_DIR, 'data', 'LCQMC.csv')
    data_loader = DataLoader(train_data_path)
    sent_models = SentMatching(data_loader)
    model = sent_models.model
    encoder = sent_models.encoder
    ranking_model = sent_models.get_ranking_model(data_loader.y_valid.shape[0])

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
        _ = new_result[:, 0] != new_result[:, 0]  # 生成一个全为False的向量
        for i in range(10):  # 注意按照相似度排序的话，第一个就是输入句子（全匹配）
            _ = _ + (new_result[:, 0] == new_result[:, i + 1])
            if i + 1 == 1:
                top1_acc = 1. * _.sum() / len(_)
            elif i + 1 == 5:
                top5_acc = 1. * _.sum() / len(_)
            elif i + 1 == 10:
                top10_acc = 1. * _.sum() / len(_)

        return top1_acc, top5_acc, top10_acc

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

    # test step
    test_data, x_test, y_test, test_id2g = data_loader.process_test_data(os.path.join(ROOT_DIR, 'data', 'collection.csv'))
    test_vec = encoder.predict(x_test,
                               verbose=True,
                               batch_size=1000)  # encoder计算句向量

    top1_acc, top5_acc, top10_acc = evaluate_test(encoder, ranking_model, test_vec, test_id2g)

    print('top1_acc: %s, top5_acc: %s, top10_acc: %s' % (top1_acc, top5_acc, top10_acc))

    def most_similar(s):
        v = encoder.predict(np.array([data_loader.string2id(strQ2B(s).lower())]))[0]
        sims = np.dot(test_vec, v)
        for i in sims.argsort()[-10:][::-1]:
            print(test_data.iloc[i][1], sims[i])

    most_similar('我下午就去还钱，好吧。')


if __name__ == '__main__':
    main()
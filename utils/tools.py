# coding=utf-8

from tqdm import tqdm
import pickle
import os

ROOT_DIR = os.getcwd()


def strQ2B(ustring):  # 全角转半角
    rstring = ''
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


def construct_dict(origin_data, min_count, save=True):
    chars = {}
    for s in tqdm(iter(origin_data)):
        for c in s:
            if c not in chars:
                chars[c] = 0
            chars[c] += 1

    # 0: padding标记
    # 1: unk标记
    chars = {i: j for i, j in chars.items() if j >= min_count}
    id2char = {i + 2: j for i, j in enumerate(chars)}
    char2id = {j: i for i, j in id2char.items()}
    if save:
        # 保存数据
        with open(os.path.join(ROOT_DIR, 'cache_data', 'char2id.pickle'), 'wb') as f:
            pickle.dump(char2id, f, -1)
    return char2id, len(chars)


if __name__ == '__main__':
    print(os.path.join(ROOT_DIR, 'cache_data', 'char_2_id.pickle'))
    print(os.path.realpath(__file__))
    print(os.path.abspath(os.path.realpath(__file__)))
import matplotlib.pyplot as plt
import numpy as np
import torch


def print_list(l, j):
    for i in range(1,j):
        print(l[-i])


class Vocab:
    def __init__(self, path):
        self.vocab = self.read_vocab(path)
        self.pad_id, self.sos_id, self.eos_id = 0, 1, 2
        self.word2ids = {item[0]: item[1] for item in self.vocab}
        self.id2words = {v: k for k, v in self.word2ids.items()}

    def read_vocab(self, path):
        vocab = []
        id = 0
        fo = open(path, 'r')
        for row in fo:
            row = row.strip('\n').split(' ')[0]
            vocab.append((row, id))
            id += 1
        return vocab

class DataProcesser:
    def __init__(self, args):

        print('Process data...')

        self.args = args
        zh_text = self.read_text(self.args.zh_data_path)
        en_text = self.read_text(self.args.en_data_path)
        self.zh_vocab = Vocab(self.args.zh_voc_path)
        self.en_vocab = Vocab(self.args.en_voc_path)
        zh_ids = self.text2id(zh_text, self.zh_vocab)
        en_ids = self.text2id(en_text, self.en_vocab)

        self.zh_inputs = self.gene_input(zh_ids)
        self.en_inputs = self.gene_input(en_ids)
        #
        self.train_idx = range(240000)
        self.valid_idx = range(240000, 250000)
        self.test_idx = range(250000, 252777)

        # self.train_idx = range(800)
        # self.valid_idx = range(100, 200)
        # self.test_idx = range(100, 200)

        print('train:{},valid:{},test:{}'.format(len(self.train_idx), len(self.valid_idx), len(self.test_idx)))

    # 读取原始文本
    def read_text(self, path):
        data_text = []
        fo = open(path, 'r')
        for row in fo:
            data_text.append(row.strip('\n'))
        fo.close()
        return data_text

    def text2id(self, texts, vocab):

        ids = []
        for text in texts:
            text = text.strip('\n').split(' ')
            delete_text = []
            for item in text:
                if len(item) != 0 and '\u2028' not in item:
                    delete_text.append(item)
            # print(delete_text)
            ids.append([vocab.word2ids[item] for item in delete_text])

        return ids

    def gene_input(self, ids):
        inputs = {'inputs': [], 'pad_masks': []}
        for id in ids:
            input = id[:min(self.args.max_length-2, len(id))]
            input = [1] + input + [2]
            input = input + [0] * (self.args.max_length-len(input))

            pad_mask = np.ones(self.args.max_length)
            pad_mask[np.array(input) == 0] = 0
            inputs['inputs'].append(input)
            inputs['pad_masks'].append(pad_mask)
        inputs['inputs'] = torch.LongTensor(inputs['inputs'])
        inputs['pad_masks'] = torch.LongTensor(inputs['pad_masks'])
        return inputs


def vision(L):
    cnt = {}
    for li in L:
        l = len(li)
        if l not in cnt:
            cnt[l] = 0
        cnt[l] += 1
    cnt = sorted(cnt.items(), key=lambda d: d[1])
    x = [k[0] for k in cnt]
    y = [k[1] for k in cnt]
    plt.bar(x, y)
    plt.show()

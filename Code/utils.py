import torch
import numpy as np
from BatchDataGenerater import *
from torch import nn
from torch.nn import functional as F
import time
import math
import collections


def bleu(pred_seq, label_seq, k): 
 
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))

    for n in range(1, k + 1):

        num_matches, label_subs = 0, collections.defaultdict(int)

        for i in range(len_label - n + 1):
            label_subs[''.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[''.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[''.join(pred_tokens[i: i + n])] -= 1

        if len_pred - n + 1 <= 0:
            continue
            
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


# encoder的mask
def subsequent_mask(q, k):
    bz, head, seq_len_q = q.shape[0], q.shape[1], q.shape[2]
    bz, head, seq_len_k = k.shape[0], k.shape[1], q.shape[2]
    attn_shape = (1, seq_len_q, seq_len_k)
    sequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    sequent_mask = torch.from_numpy(sequent_mask) == 0
    sequent_mask = sequent_mask.long().cuda()
    sequent_mask = sequent_mask.expand(bz, seq_len_q, seq_len_k)
    sequent_mask = sequent_mask.unsqueeze(1)
    sequent_mask = sequent_mask.expand(bz, head, seq_len_q, seq_len_k)
    return sequent_mask


def pad_mask(q, k, mask):
    '''
    :param q: batch*head*seq_length*hidden_dim
    :param k: batch*head*seq_length*hidden_dim
    :return:
    '''
    bz, head, seq_len_q = q.shape[0], q.shape[1], q.shape[2]
    bz, head, seq_len_k = k.shape[0], k.shape[1], k.shape[2]
    # print(bz, head, seq_len_q, seq_len_k, mask.shape)
    pad_att_mask = mask.unsqueeze(1)
    pad_att_mask = pad_att_mask.expand(bz, seq_len_q, seq_len_k)
    pad_att_mask = pad_att_mask.unsqueeze(1)
    pad_att_mask = pad_att_mask.expand(bz, head, seq_len_q, seq_len_k)
    return pad_att_mask


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred, label, mask):
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * mask).mean(dim=1)
        return weighted_loss.sum()


def train(args, model, Criterion, Optimizer, data):
    model.train()

    for epoch in range(args.epochs):
        print('start training epoch:{}...'.format(epoch))
        batch_loss, batch_time = batch_train(args, model, Criterion, Optimizer, data)

        print('epoch:{},loss:{},time:{}'.format(epoch, batch_loss, batch_time))

        eval(args, model, data)
        inferencer(args, model, data)


def batch_train(args, model, Criterion, Optimizer, data):
    start_time = time.time()
    data_gene = BatchDataGenerater(args, data.zh_inputs, data.en_inputs, data.train_idx)
    all_loss = 0

    for src_inputs, src_pad_masks, tar_inputs, tar_pad_masks, tar_labels in data_gene:
        Optimizer.zero_grad()
        out = model(src_inputs, tar_inputs, src_pad_masks, tar_pad_masks)
        loss = Criterion(out, tar_labels, tar_pad_masks)
        all_loss += loss
        loss.backward()
        Optimizer.step()
    del data_gene
    return all_loss / len(data.train_idx), time.time() - start_time


@torch.no_grad()
def eval(args, model, data):
    print('start eval valid...')
    model.eval()
    start_time = time.time()
    data_gene = BatchDataGenerater(args, data.zh_inputs, data.en_inputs, data.valid_idx)
    all_bleu = 0
    for src_inputs, src_pad_masks, tar_inputs, tar_pad_masks, tar_labels in data_gene:
        tar_label = tar_labels
        pred_seqs = beam_search(args, model, src_inputs, src_pad_masks)
        tar_seqs = tar_label.tolist()
        for i in range(src_inputs.shape[0]):
            pred_seq = pred_seqs[i]
            tar_seq = tar_seqs[i]

            sentence_to = decode_sentence(pred_seq, data.en_vocab, 'en')
            sentence_to_label = decode_sentence(tar_seq, data.en_vocab, 'en')

            sentence_to = delete_other_token_from_sentence(sentence_to)
            sentence_to_label = delete_other_token_from_sentence(sentence_to_label)

            _bleu = bleu(sentence_to, sentence_to_label, 4)
            all_bleu += _bleu

    print('valid bleu:{},time:{}'.format(all_bleu/len(data.valid_idx), time.time()-start_time))


def decode_sentence(seq, vocab, lang='en'):
    words = []
    for word_idx in seq:
        words.append(vocab.id2words[word_idx])
    sentence = ''
    if lang == 'en':
        sentence = ' '.join(words)
        sentence = sentence.replace('@@ ', '')
        sentence = sentence.replace('@@', '')
    if lang == 'zh':
        sentence = ''.join(words)
        sentence = sentence.replace('@', '')
    return sentence


def delete_other_token_from_sentence(sentence):

    if '<BOS>' in sentence:
        sentence = sentence[5:]
    if '<EOS>' in sentence:
        pos_eos = sentence.index('<EOS>')
        sentence = sentence[:pos_eos]
    return sentence


@torch.no_grad()
def inferencer(args, model, data):
    print('start inference...')
    data_gene = BatchDataGenerater(args, data.zh_inputs, data.en_inputs, data.test_idx)
    cnt = 0
    for src_inputs, src_pad_masks, tar_inputs, tar_pad_masks, tar_labels in data_gene:
        src_label = src_inputs
        tar_label = tar_labels
        pred_seqs = beam_search(args, model, src_inputs, src_pad_masks)
        src_seqs = src_label.tolist()
        tar_seqs = tar_label.tolist()
        for i in range(src_inputs.shape[0]):
            src_seq = src_seqs[i]
            pred_seq = pred_seqs[i]
            tar_seq = tar_seqs[i]

            sentence_from = decode_sentence(src_seq, data.zh_vocab, 'zh')
            sentence_to = decode_sentence(pred_seq, data.en_vocab, 'en')
            sentence_to_label = decode_sentence(tar_seq, data.en_vocab, 'en')

            sentence_from = delete_other_token_from_sentence(sentence_from)
            sentence_to = delete_other_token_from_sentence(sentence_to)
            sentence_to_label = delete_other_token_from_sentence(sentence_to_label)

            print(sentence_from+' ==> '+sentence_to_label)
            print(sentence_from+' ==> '+sentence_to)
            print('---------------------------------------------------------------------------------------------------')

            if args.inference_count < cnt:
                return
            cnt += 1


# batch beam search
@torch.no_grad()
def get_init_state(args, model, src_inputs, src_pad_masks):
    batch_size = src_inputs.shape[0]
    beam_size = args.beam_size

    encode_out = model.encode(src_inputs, src_pad_masks)

    # bz * 1
    init_tar = torch.tensor([[1]] * batch_size).cuda()  # 1 is bos #以bos开始翻译

    # bz*seq_len*dim
    decode_out = model.decode(encode_out, init_tar, src_pad_masks, None)  # 获得decode的最后一个时间步的输出

    # bz* voc_sz
    decode_out = F.softmax(decode_out, dim=-1)[:, -1, :]

    # bz * beam_sz, bz * beam_sz
    best_k_probs, best_k_idx = decode_out.topk(beam_size)  # 1*beam_size, 1*beam_size 挑选出前k个

    # bz * beam_sz
    scores = torch.log(best_k_probs).view(batch_size, beam_size)  # beam_size

    # bz * beam_sz * seq_len
    tar_seq = torch.full((batch_size, beam_size, args.max_length), 0)  # 构造beam_size个 目标序列 beam_size*seq_length
    tar_seq[:, :, 0] = 1  # 第一位为bos
    tar_seq[:, :, 1] = best_k_idx  # 第二位为 刚找出的最大的beam_size个

    # bz*beam_sz*seq_len* dim
    encode_out = encode_out.unsqueeze(1).repeat(1, beam_size, 1, 1)  # 构造三个encode输出 beam_size*seq_length*dim

    # bz * beam_sz * seq_len * dim
    # bz * beam_sz * seq_len
    # bz * beam_sz

    return encode_out, tar_seq, scores


def get_the_best_score_and_idx(args, tar_seq, decode_out, scores, step):
    batch_size = scores.shape[0]
    beam_size = args.beam_size

    # bz*beam_size*beam_size
    best_k2_probs, best_k2_idx = decode_out.topk(beam_size)

    # bz*beam_size*beam_size=bz*beam_size*beam_size + bz*beam_size*1
    scores = torch.log(best_k2_probs).view(batch_size, beam_size, -1) + scores.view(batch_size, beam_size, 1)  # 累加分数

    # bz*beam_size
    scores, best_k_idx_in_k2 = scores.view(batch_size, -1).topk(beam_size)  # 获得当前topk的分数和其idx

    # bz*beam_size
    best_k_r_idx, best_c_idx = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size  # 获得topk在原始beam_size*beam_size矩阵的行和列

    # bz*beam_size()
    best_k_idx = torch.zeros(batch_size, beam_size)
    for i in range(batch_size):
        best_k_idx[i] = best_k2_idx[i][best_k_r_idx[i], best_c_idx[i]]
    best_k_idx = best_k_idx.long()

    # bz*beam_size*seq_len
    tar_seq = tar_seq.view(batch_size, beam_size, -1)
    for i in range(batch_size):
        tar_seq[i][:, :step] = tar_seq[i][best_k_r_idx[i], :step]  # 将上一步的序列更换为本次分数最高的

    tar_seq[:, :, step] = best_k_idx  # 当前step的topk

    # bz*beam_size*seq_len
    # bz*beam_size
    return tar_seq, scores


@torch.no_grad()
def beam_search(args, model, src_inputs, src_pad_masks):

    beam_size = args.beam_size
    encode_out, tar_seq, scores = get_init_state(args, model, src_inputs, src_pad_masks)

    src_pad_masks = src_pad_masks.unsqueeze(1).repeat(1, beam_size, 1)

    # 保存答案
    res_seq = [[] for _ in range(args.batch_size)]

    # 删除后进行新batch_idx到原始batch_idx的映射
    map_idx = np.array([i for i in range(src_inputs.shape[0])])

    for step in range(2, args.max_length):
        batch_size = scores.shape[0]

        tar_seq = tar_seq.cuda()
        encode_out = encode_out.view(-1, encode_out.shape[-2], encode_out.shape[-1])
        src_pad_masks = src_pad_masks.view(-1, src_pad_masks.shape[-1])
        tar_seq = tar_seq.view(-1, tar_seq.shape[-1])

        decode_out = model.decode(encode_out, tar_seq, src_pad_masks, None)

        # bz * beam_size * seq_len * dim
        decode_out = decode_out.view(batch_size, beam_size, decode_out.shape[-2], decode_out.shape[-1])

        decode_out = F.softmax(decode_out, dim=-1)[:, :, -1, :]
        tar_seq, scores = get_the_best_score_and_idx(args, tar_seq, decode_out, scores, step)

        encode_out = encode_out.view(batch_size, -1, encode_out.shape[-2], encode_out.shape[-1])
        src_pad_masks = src_pad_masks.view(batch_size, -1, src_pad_masks.shape[-1])

        # 查看是否生成 eos
        # bz*beam_size*seq_len
        eos_locs = tar_seq == 2  # 2 is eos

        # 构造[1, 2, 3, ... , max_length]的tensor
        # bz*1*seq_len
        len_map = torch.arange(1, args.max_length + 1, dtype=torch.long). \
            unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1).cuda()

        # 获取生成的句子的长度（生成了eos），对于不是eos的点，全部mask为 max_length, 所以只有除了eos的位置，其他地方都为max_length
        # 这样在对行取最小值，即为生成句子的长度
        # bz*beam_size
        # 对应每个batch里beam_size个候选项的seq长度
        seq_lens, _ = len_map.masked_fill(~eos_locs, args.max_length).min(2)

        # 检查每个batch， 生成成功的话记录下答案，并在生成任务中删除该项
        ans_got = set()
        idx_got = set()
        for i in range(batch_size):
            ans_idx = 0
            # 如果所有的句子都生成了eos
            if (eos_locs[i].sum(-1) > 0).sum(0).item() == beam_size:
                # 选择分数最高的那个句子，并且对短句子进行惩罚
                _, ans_idx = scores[i].div(seq_lens[i].float() ** args.len_penalty).max(0)
                ans_idx = ans_idx.item()
                res_seq[map_idx.tolist().index(i)] = tar_seq[i][ans_idx][:seq_lens[i][ans_idx]].tolist()
                ans_got.add(i)
                idx_got.add(map_idx.tolist().index(i))
        # 删除生成完毕的句子
        for idx in idx_got:
            map_idx[idx:] -= 1
            map_idx[idx] = -1
        encode_out = remove_tensor(encode_out, ans_got)
        tar_seq = remove_tensor(tar_seq, ans_got)
        src_pad_masks = remove_tensor(src_pad_masks, ans_got)
        scores = remove_tensor(scores, ans_got)

        if len(encode_out) == 0: break

    # 对于没有找到答案的，选择其第一个
    for i, item in enumerate(map_idx):
        if item >= 0:
            res_seq[i] = tar_seq[item][0][:seq_lens[item][0]].tolist()

    return res_seq


# 删除已经找到答案的部分，减少运算量
def remove_tensor(tensor, remove_set):
    deleted = 0
    for i in remove_set:
        now_del = i - deleted
        tensor = torch.cat((tensor[0:now_del], tensor[now_del + 1:]), dim=0)
        deleted += 1
    return tensor

# single beam_search

# @torch.no_grad()
# def get_init_state(args, model, src_inputs, src_pad_masks):
#
#     beam_size = args.beam_size
#
#     encode_out = model.encode(src_inputs, src_pad_masks)
#
#     init_tar = torch.tensor([[[1]]]).cuda() # 1 is bos #以bos开始翻译
#
#     decode_out = model.decode(encode_out, init_tar, src_pad_masks, None)# 获得decode的最后一个时间步的输出
#     decode_out = F.softmax(decode_out, dim=-1)[:, -1, :]
#     best_k_probs, best_k_idx = decode_out.topk(beam_size) # 1*beam_size, 1*beam_size 挑选出前k个
#     scores = torch.log(best_k_probs).view(beam_size) # beam_size
#     tar_seq = torch.full((beam_size, args.max_length), 0) #构造beam_size个 目标序列 beam_size*seq_length
#     tar_seq[:, 0] = 1 # 第一位为bos
#     tar_seq[:, 1] = best_k_idx #第二位为 刚找出的最大的beam_size个
#     encode_out = encode_out.repeat(beam_size, 1, 1) # 构造三个encode输出 beam_size*seq_length*dim
#
#     return encode_out, tar_seq, scores
#
#
# def get_the_best_score_and_idx(args, tar_seq, decode_out, scores, step):
#     beam_size = args.beam_size
#     best_k2_probs, best_k2_idx = decode_out.topk(beam_size) # beam_size * beam_size
#
#     scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1) #累加分数
#     scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size) # 获得当前topk的分数和其idx
#     best_k_r_idx, best_c_idx = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 // beam_size # 获得topk在原始beam_size*beam_size矩阵的行和列
#     best_k_idx = best_k2_idx[best_k_r_idx, best_c_idx] # 获得topk在词典中的idx
#
#     tar_seq[:, :step] = tar_seq[best_k_r_idx, :step] # 将上一步的序列更换为本次分数最高的
#     tar_seq[:, step] = best_k_idx # 当前step的topk
#     return tar_seq, scores
#
#
# @torch.no_grad()
# def beam_search(args, model, src_inputs, src_pad_masks):
#     src_inputs = expand_dim(src_inputs)
#     src_pad_masks = expand_dim(src_pad_masks)
#
#     beam_size = args.beam_size
#     encode_out, tar_seq, scores = get_init_state(args, model, src_inputs, src_pad_masks)
#
#     ans_idx = 0
#
#     for step in range(1, args.max_length):
#
#         tar_seq = tar_seq.cuda()
#         decode_out = model.decode(encode_out, tar_seq, src_pad_masks, None)
#         decode_out = F.softmax(decode_out, dim=-1)[:, -1, :]
#         tar_seq, scores = get_the_best_score_and_idx(args, tar_seq, decode_out, scores, step)
#         # 查看是否生成 eos
#         eos_locs = tar_seq == 2 # 2 is eos
#         #构造[1, 2, 3, ... , max_length]的tensor
#         len_map = torch.arange(1, args.max_length + 1, dtype=torch.long).unsqueeze(0).cuda()
#
#         # 获取生成的句子的长度（生成了eos），对于不是eos的点，全部mask为 max_length, 所以只有除了eos的位置，其他地方都为max_length
#         # 这样在对行取最小值，即为生成句子的长度。
#         seq_lens, _ = len_map.masked_fill(~eos_locs, args.max_length).min(1)
#
#         # 如果所有的句子都生成了eos
#         if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
#             # 选择分数最高的那个句子，并且对短句子进行惩罚
#             _, ans_idx = scores.div(seq_lens.float() ** args.len_penalty).max(0)
#             ans_idx = ans_idx.item()
#
#     return tar_seq[ans_idx][:seq_lens[ans_idx]].tolist()



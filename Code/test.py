import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
#
# a = torch.tensor([
#     [0,0,1],
#     [0,0,1]
# ])
# a = a.unsqueeze(1)
# print(a)
# print(a.size())
# b = a.expand(2,3,3)
#
# print(b)
# print(b.size())
# b = b.unsqueeze(1)
# print(b.size())
# c = torch.randn(2,4,3,3)
# b = b.expand(2,4,3,3)
# c.masked_fill_(b, 0.01)
# print(c.size())
# def subsequent_mask(size):
#     attn_shape = (1, size, size)
#     subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
#     return torch.from_numpy(subsequent_mask) == 0
#
# a = subsequent_mask(5)
# print(a)
# a = torch.tensor([1,0,0])
# b = torch.tensor([1,0,1])
# print(a&b)
#
# a = np.array([])

# from nltk.translate.bleu_score import sentence_bleu
# reference = [[1, 2, 3, 4, 7, 6]]
# candidate = [1, 2, 3, 4, 5, 6, 6, 6, 6]
# score = sentence_bleu(reference, candidate)
# print(score)
# out = torch.tensor([[
#     [1.,2.2,3.],
#     [4.,5.,6.]
# ],[
#     [1.,2.2,3.],
#     [4.,5.,6.]
# ]])
# b = torch.tensor([1,1,1,0]).tolist()
# print(b.index(0))
# print(b[:3])
# print(out.shape)
# a = out.argmax(-1)
# print(a)
# print(a.shape)
a = '他@@认为这@@最终会@@增加@@国家@@收入@@，因为@@这种@@直接@@刺激@@的'
print(a.replace('@@',''))
# a = torch.tensor([[1,2,2],[1,3,3],[1,3,1]])
# b = torch.tensor([1,2,3]).view(3,1)
# c = a+b
# eos_locs = a == 3
# print(~eos_locs)
# print(eos_locs.sum(1))
# print(a.shape)
# b = a.unsqueeze(0)
# print(b)
# print(b.shape)
# b = torch.tensor([1,2])
# b = b.unsqueeze(-1)
# print(b.shape)
# print(b)
# c = torch.cat((a, b),dim=-1)
# print(c)
# a = torch.randn((2, 3, 4))
# print(a.shape)
# print(a)
# print('-----------------')
# a = a.view(-1, 4)
# print(a.shape)
# print(a)
# print('-----------------')
# a = a.view(2, 3 ,4)
# print(a.shape)
# print(a)
# print('-----------------')
# a = torch.randn(2,3,3)
# print(a)
# b = torch.tensor([[0,1,0],[1,0,0]])
# c = torch.tensor([[1,2,1],[1,2,0]])
# res = torch.zeros(2, 3)
# for i in range(2):
#     res[i] = a[i][b[i],c[i]]
#
#
#
#


a = torch.randn(5,3,4)
b = set([0,1,2,3,4])
pre = 0
print(a)
for i in b:
    now_del = i-pre
    a = torch.cat((a[0:now_del], a[now_del+1:]), dim=0)
    pre += 1

print(len(a))
print(a.shape)

#
#
#
#
#
#
#
#
#
#
# @torch.no_grad()
# def get_init_state(args, model, src_inputs, src_pad_masks):
#
#     batch_size = src_inputs.shape[0]
#     beam_size = args.beam_size
#
#     encode_out = model.encode(src_inputs, src_pad_masks)
#
#     # bz * 1
#     init_tar = torch.tensor([[1]]*batch_size).cuda() # 1 is bos #以bos开始翻译
#
#     # bz*seq_len*dim
#     decode_out = model.decode(encode_out, init_tar, src_pad_masks, None)# 获得decode的最后一个时间步的输出
#
#     # bz* voc_sz
#     decode_out = F.softmax(decode_out, dim=-1)[:, -1, :]
#
#     # bz * beam_sz, bz * beam_sz
#     best_k_probs, best_k_idx = decode_out.topk(beam_size) # 1*beam_size, 1*beam_size 挑选出前k个
#
#     # bz * beam_sz
#     scores = torch.log(best_k_probs).view(batch_size, beam_size) # beam_size
#
#     # bz * beam_sz * seq_len
#     tar_seq = torch.full((batch_size, beam_size, args.max_length), 0) #构造beam_size个 目标序列 beam_size*seq_length
#     tar_seq[:, :, 0] = 1 # 第一位为bos
#     tar_seq[:, :, 1] = best_k_idx #第二位为 刚找出的最大的beam_size个
#
#     # bz*beam_sz*seq_len* dim
#     encode_out = encode_out.unsqueeze(1).repeat(1, beam_size, 1, 1) # 构造三个encode输出 beam_size*seq_length*dim
#
#     # bz * beam_sz * seq_len * dim
#     # bz * beam_sz * seq_len
#     # bz * beam_sz
#
#     return encode_out, tar_seq, scores
#
#
# def get_the_best_score_and_idx(args, tar_seq, decode_out, scores, step):
#     batch_size = tar_seq.shape[0]
#
#     beam_size = args.beam_size
#
#     # bz*beam_size*beam_size
#     best_k2_probs, best_k2_idx = decode_out.topk(beam_size)
#
#     # bz*beam_size*beam_size=bz*beam_size*beam_size + bz*beam_size*1
#     scores = torch.log(best_k2_probs).view(batch_size, beam_size, -1) + scores.view(batch_size, beam_size, 1) #累加分数
#
#     # bz*beam_size
#     scores, best_k_idx_in_k2 = scores.view(batch_size, -1).topk(beam_size) # 获得当前topk的分数和其idx
#
#     # bz*beam_size
#     best_k_r_idx, best_c_idx = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size # 获得topk在原始beam_size*beam_size矩阵的行和列
#
#     # bz*beam_size()
#     best_k_idx = torch.zeros(batch_size, beam_size)
#     for i in range(batch_size):
#         best_k_idx[i] = best_k2_idx[i][best_k_r_idx[i],best_c_idx[i]]
#     best_k_idx = best_k_idx.long()
#
#     # bz*beam_size*seq_len
#     tar_seq = tar_seq.view(batch_size, beam_size, -1)
#     for i in range(batch_size):
#         tar_seq[i][:, :step] = tar_seq[i][best_k_r_idx[i], :step] # 将上一步的序列更换为本次分数最高的
#
#     tar_seq[:, :, step] = best_k_idx # 当前step的topk
#
#     # bz*beam_size*seq_len
#     # bz*beam_size
#     return tar_seq, scores
#
#
# @torch.no_grad()
# def beam_search(args, model, src_inputs, src_pad_masks):
#     # src_inputs = expand_dim(src_inputs)
#     # src_pad_masks = expand_dim(src_pad_masks)
#     beam_size = args.beam_size
#     encode_out, tar_seq, scores = get_init_state(args, model, src_inputs, src_pad_masks)
#
#     src_pad_masks = src_pad_masks.unsqueeze(0).repeat(1, beam_size, 1).view(-1, src_pad_masks.shape[-1])
#     ans_idx = 0
#
#     for step in range(2, args.max_length):
#         batch_size = tar_seq.shape[0]
#         tar_seq = tar_seq.cuda()
#         encode_out = encode_out.view(-1, encode_out.shape[-2], encode_out.shape[-1])
#         tar_seq = tar_seq.view(-1, tar_seq.shape[-1])
#
#         decode_out = model.decode(encode_out, tar_seq, src_pad_masks, None)
#
#         # bz * beam_size * seq_len * dim
#         decode_out = decode_out.view(batch_size, beam_size, decode_out.shape[-2], decode_out.shape[-1])
#
#         decode_out = F.softmax(decode_out, dim=-1)[:, :, -1, :]
#         tar_seq, scores = get_the_best_score_and_idx(args, tar_seq, decode_out, scores, step)
#
#
#         tar_seq[0][1][3] = 2
#         tar_seq[0][2][5] = 2
#         tar_seq[1][0][4] = 2
#
#
#         # 查看是否生成 eos
#         # bz*beam_size*seq_len
#         eos_locs = tar_seq == 2 # 2 is eos
#
#         # 构造[1, 2, 3, ... , max_length]的tensor
#         # bz*1*seq_len
#         len_map = torch.arange(1, args.max_length + 1, dtype=torch.long).\
#             unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
#
#         # 获取生成的句子的长度（生成了eos），对于不是eos的点，全部mask为 max_length, 所以只有除了eos的位置，其他地方都为max_length
#         # 这样在对行取最小值，即为生成句子的长度
#         # bz*beam_size
#         # 对应每个batch里beam_size个候选项的seq长度
#         seq_lens, _ = len_map.masked_fill(~eos_locs, args.max_length).min(2)
#
#         # 如果所有的句子都生成了eos
#         if (eos_locs.sum() > 0).sum(0).item() == beam_size:
#             # 选择分数最高的那个句子，并且对短句子进行惩罚
#             _, ans_idx = scores.div(seq_lens.float() ** args.len_penalty).max(0)
#             ans_idx = ans_idx.item()
#
#     return tar_seq[ans_idx][:seq_lens[ans_idx]].tolist()

a = np.array([0, 1, 2, 3, 4])
a[2:] -= 1
a[2] = -1
print(a)
print(a.tolist().index(2))
for i, a in enumerate(a):
    print(i,a)

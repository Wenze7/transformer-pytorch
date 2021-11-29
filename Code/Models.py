import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import copy
from utils import *


def positional_encoding(input_x, dim, max_length=100):
    '''

    :param input_x: 输入的句子 batch*seq_length*dim
    :param dim: 每个词的维度
    :param max_length: 句子的最大长度
    :return: 位置编码后的向量 batch*word_num*dim
    '''
    # 初始化位置编码为0
    pos_emb = torch.zeros((1, max_length, dim))
    value = torch.arange(max_length, dtype=torch.float32).reshape(-1, 1) / torch.pow(
        10000,
        torch.arange(0, dim, 2, dtype=torch.float32) / dim)

    pos_emb[:, :, 0::2] = torch.sin(value).cuda() # 0,2,4...的pos_emb
    pos_emb[:, :, 1::2] = torch.cos(value).cuda()# 1,3,5...的pos_emb
    # print(input_x.shape)
    # print(pos_emb[:, :input_x.shape[1], :].shape)

    output_x = input_x + pos_emb[:, :input_x.shape[1], :].cuda()

    return output_x


# 产生N个完全相同的网络层
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):

    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()
        self.args = args

        # 定义三个转换矩阵
        self.q_linear = nn.Linear(self.args.dim, self.args.dim)
        self.w_linear = nn.Linear(self.args.dim, self.args.dim)
        self.v_linear = nn.Linear(self.args.dim, self.args.dim)

        # 多头注意力拼接之后的转换矩阵
        self.w_0_linear = nn.Linear(self.args.dim, self.args.dim)

        self.dropout = nn.Dropout(self.args.dropout)

        # Q*K之后的缩放
        self.scale = torch.sqrt(torch.FloatTensor([self.args.dim // self.args.num_heads])).cuda()


    def forward(self, q, k, v, mask=None, encoder = False):
        Q = self.q_linear(q)
        K = self.w_linear(k)
        V = self.v_linear(v)

        bsz = Q.shape[0]

        # 拆分多头注意力 [batch_size, seq_length, dim] - > [batch_size, num_heads, seq_length, dim/num_heads]
        # example [10, 5, 60] -> [10, 6 , 5, 10]
        Q = Q.view(bsz, -1, self.args.num_heads, self.args.dim // self.args.num_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.args.num_heads, self.args.dim // self.args.num_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.args.num_heads, self.args.dim // self.args.num_heads).permute(0, 2, 1, 3)

        # Q, K 相乘计算注意力分数
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # mask掉不需要的attention
        last_mask = None
        pad_att_mask = None
        if mask is not None:
            pad_att_mask = pad_mask(Q, K, mask)

        sequent_mask = None
        if encoder == True:
            sequent_mask = subsequent_mask(Q, K)

        if encoder == False:
            last_mask = pad_att_mask
        else:
            if pad_att_mask == None:
                last_mask = sequent_mask
            else:
                last_mask = sequent_mask | pad_att_mask

        attention = attention.masked_fill(last_mask == 0, -1e10)

        # Softmax
        attention = torch.softmax(attention, dim=-1)

        # attention 和 V 相乘
        out = torch.matmul(attention, V)

        # 下面是多头注意力的拼接
        out = out.permute(0, 2, 1, 3).contiguous().view(bsz, -1, self.args.dim)
        attention_out = self.w_0_linear(out)

        # dropout
        attention_out = self.dropout(attention_out)

        return attention_out


# 每一层后面的前馈层
class FeedForward(nn.Module):
    def __init__(self, args):
        super(FeedForward, self).__init__()
        self.args = args
        self.activate = nn.ReLU()
        self.linear_1 = nn.Linear(self.args.dim, self.args.ff_dim)
        self.linear_2 = nn.Linear(self.args.ff_dim, self.args.dim)
        self.dropout = nn.Dropout(self.args.dropout)

    def forward(self, x):
        return self.linear_2(
            self.dropout(self.activate(self.linear_1(x)))
        )


# 层归一化 LayerNorm
class LayerNorm(nn.Module):
    def __init__(self, args):
        super(LayerNorm, self).__init__()
        self.args = args
        self.a_2 = nn.Parameter(torch.ones(self.args.dim))
        self.b_2 = nn.Parameter(torch.zeros(self.args.dim))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.args.eps) + self.b_2


# 残差链接
class AddAndNorm(nn.Module):
    def __init__(self, args):
        super(AddAndNorm, self).__init__()
        self.args = args
        self.norm = LayerNorm(copy.deepcopy(self.args))
        self.dropout = nn.Dropout(self.args.dropout)

    def forward(self, x):
        return x + self.dropout(self.norm(x))


# 单个EncoderLayer
class EncoderLayer(nn.Module):
    def __init__(self, args):
        super(EncoderLayer, self).__init__()
        self.args = args
        self.MultiHeadAttention = MultiHeadAttention(copy.deepcopy(self.args))
        self.FeedForward = FeedForward(copy.deepcopy(self.args))
        self.AddAndNorms = clones(AddAndNorm(copy.deepcopy(self.args)), 2)

    def forward(self, x, mask):
        out = self.AddAndNorms[0](self.MultiHeadAttention(x, x, x, mask))
        out = self.AddAndNorms[1](self.FeedForward(out))
        return out


# 单个DecoderLayer
class DecoderLayer(nn.Module):
    def __init__(self, args):
        super(DecoderLayer, self).__init__()
        self.args = args
        self.SelfMultiHeadAttention = MultiHeadAttention(copy.deepcopy(self.args))
        self.CrossMultiHeadAttention = MultiHeadAttention(copy.deepcopy(self.args))
        self.FeedForward = FeedForward(copy.deepcopy(self.args))
        self.AddAndNorms = clones(AddAndNorm(copy.deepcopy(self.args)), 3)

    def forward(self, src_x, tar_x, src_mask, tar_mask):
        tar_x = self.AddAndNorms[0](self.SelfMultiHeadAttention(tar_x, tar_x, tar_x, tar_mask, encoder=True))
        tar_x = self.AddAndNorms[1](self.CrossMultiHeadAttention(tar_x, src_x, src_x, src_mask))
        out = self.AddAndNorms[1](self.FeedForward(tar_x))
        return out


# Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        self.args = args
        self.emb = nn.Embedding(self.args.src_voc_size, self.args.dim)
        self.layers = clones(EncoderLayer(copy.deepcopy(self.args)), self.args.num_stacks)

    def forward(self, x, mask):
        x = self.emb(x).cuda()
        x = x * math.sqrt(self.args.dim)
        x = positional_encoding(x, self.args.dim)
        for encoder_layer in self.layers:
            x = encoder_layer(x, mask)
        return x


# Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, args):
        super(TransformerDecoder, self).__init__()
        self.args = args
        self.emb = nn.Embedding(self.args.tar_voc_size, self.args.dim)
        self.layers = clones(DecoderLayer(copy.deepcopy(self.args)), self.args.num_stacks)

    def forward(self, src_x, tar_x, src_mask, tar_mask):
        # print(tar_x)
        tar_x = self.emb(tar_x).cuda()
        tar_x = tar_x * math.sqrt(self.args.dim)
        tar_x = positional_encoding(tar_x, self.args.dim)

        for decoder_layer in self.layers:
            x = decoder_layer(src_x, tar_x, src_mask, tar_mask)
        return x


# 完整模型
class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.args = args
        self.encoder = TransformerEncoder(copy.deepcopy(args))
        self.decoder = TransformerDecoder(copy.deepcopy(args))
        self.Linear = nn.Linear(self.args.dim, self.args.vocab_size)
        self.weight_init()

    def forward(self, src_x, tar_x, src_mask, tar_mask):
        src_out = self.encode(src_x, src_mask)
        return self.decode(src_out, tar_x, src_mask, tar_mask)

    def encode(self, src_x, src_mask):
        return self.encoder(src_x, src_mask)

    def decode(self, src_out, tar_x, src_mask, tar_mask):
        tar_out = self.decoder(src_out, tar_x, src_mask, tar_mask)
        return self.Linear(tar_out)

    def weight_init(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

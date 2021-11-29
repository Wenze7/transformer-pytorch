import argparse
from DataProcesser import *
from Models import *
import torch
from torch import nn
from BatchDataGenerater import *
from utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=300, help='每个词的维度')
    parser.add_argument('--ff_dim', type=int, default=300, help='前馈层隐层维度')
    parser.add_argument('--num_heads', type=int, default=6, help='注意力头数')
    parser.add_argument('--num_stacks', type=int, default=6, help='encoder和decoder堆叠的层数')
    parser.add_argument('--vocab_size', type=int, default=10256, help='词典大小')
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--zh_data_path', type=str, default='../Data/zh.txt')
    parser.add_argument('--en_data_path', type=str, default='../Data/en.txt')
    parser.add_argument('--zh_voc_path', type=str, default='../Data/voc_zh.txt')
    parser.add_argument('--en_voc_path', type=str, default='../Data/voc_en.txt')
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--src_voc_size', type=int, default=15895)
    parser.add_argument('--tar_voc_size', type=int, default=10256)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--epochs', type=input, default=100)
    parser.add_argument('--beam_size', type=int, default=3)
    parser.add_argument('--len_penalty', type=float, default=0.7)
    parser.add_argument('--bleu_gram', type=int, default=4)
    parser.add_argument('--inference_count', type=int, default=5)
    args = parser.parse_args()


    data = DataProcesser(args)
    model = Transformer(args)
    # model = nn.DataParallel(model)
    model = model.cuda()
    Criterion = MaskedSoftmaxCELoss()
    # Criterion = nn.CrossEntropyLoss()
    Optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train(args, model, Criterion, Optimizer, data)


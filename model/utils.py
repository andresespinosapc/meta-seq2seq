import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attn(nn.Module):
    # batch attention module

    def __init__(self):
        super(Attn, self).__init__()

    def forward(self, Q, K, V):
        #
        # Input
        #  Q : Matrix of queries; batch_size x n_queries x query_dim
        #  K : Matrix of keys; batch_size x n_memory x query_dim
        #  V : Matrix of values; batch_size x n_memory x value_dim
        #
        # Output
        #  R : soft-retrieval of values; batch_size x n_queries x value_dim
        #  attn_weights : soft-retrieval of values; batch_size x n_queries x n_memory
        query_dim = torch.tensor(float(Q.size(2)))
        if Q.is_cuda: query_dim = query_dim.cuda()
        attn_weights = torch.bmm(Q,K.transpose(1,2)) # batch_size x n_queries x n_memory
        attn_weights = torch.div(attn_weights, torch.sqrt(query_dim))
        attn_weights = F.softmax(attn_weights, dim=2) # batch_size x n_queries x n_memory
        R = torch.bmm(attn_weights,V) # batch_size x n_queries x value_dim
        return R, attn_weights

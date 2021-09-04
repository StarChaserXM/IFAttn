# coding:utf-8
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('.')
# 多头自注意力
from tiknib.CodeAttention.attention.SelfAttention import ScaledDotProductAttention
# 外部注意力
from tiknib.CodeAttention.attention.ExternalAttention import ExternalAttention


# # 点乘注意力机制
# class ScaledDotProductAttention(nn.Module):
#     ''' Scaled Dot-Product Attention '''
#
#     def __init__(self, temperature, attn_dropout=0.1):
#         super().__init__()
#         self.temperature = temperature
#         self.dropout = nn.Dropout(attn_dropout)
#
#     def forward(self, q, k, v, mask=None):
#         attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
#
#         if mask is not None:
#             attn = attn.masked_fill(mask == 0, -1e9)
#
#         attn = self.dropout(F.softmax(attn, dim=-1))
#         output = torch.matmul(attn, v)
#
#         return output, attn
#
#
# # 多头注意力机制
# class MultiHeadAttention(nn.Module):
#     ''' Multi-Head Attention module '''
#
#     def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
#         super().__init__()
#
#         self.n_head = n_head
#         self.d_k = d_k
#         self.d_v = d_v
#
#         self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
#         self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
#         self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
#         self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
#
#         self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
#
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
#
#     def forward(self, q, k, v, mask=None):
#         d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
#         sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
#
#         residual = q
#
#         # Pass through the pre-attention projection: b x lq x (n*dv)
#         # Separate different heads: b x lq x n x dv
#         q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
#         k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
#         v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
#
#         # Transpose for attention dot product: b x n x lq x dv
#         q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
#
#         if mask is not None:
#             mask = mask.unsqueeze(1)  # For head axis broadcasting.
#
#         q, attn = self.attention(q, k, v, mask=mask)
#
#         # Transpose to move the head dimension back: b x lq x n x dv
#         # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
#         q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
#         q = self.dropout(self.fc(q))
#         q += residual
#
#         # q = self.layer_norm(q)
#
#         return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)
        return x


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, att_type, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.att_type = att_type
        self.slf_attn = ScaledDotProductAttention(d_model=1, d_k=d_k, d_v=d_v, h=n_head, dropout=dropout)
        self.ext_attn = ExternalAttention(d_model=1,S=d_k,dropout=dropout)
        # self.slf_attn = MultiHeadAttention(n_head, 1, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        # 在最后一维增加一个维度 [b,f_d] --> [b,f_d,1]
        enc_input = enc_input.unsqueeze(-1)
        if self.att_type == 'SelfAttention':
            enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, attention_mask=slf_attn_mask)
        elif self.att_type == 'ExternalAttention':
            enc_output, enc_slf_attn = self.ext_attn(enc_input)
        elif self.att_type == 'NoAttention':
            enc_output = enc_input
            enc_slf_attn = None
        # 减少最后一维 [b,f_d,1] --> [b,f_d]
        enc_output = enc_output.squeeze(-1)
        # enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


# 模型定义
# input1:[b,f_d]
# input2:[b,f_d]
# label:[b, ]
class SiameseAttentionNet(nn.Module):
    def __init__(self, feature_dim, hidden_dim, n_layers, n_head, d_k, d_v, att_type, dropout):
        super().__init__()
        self.layer_stack = nn.ModuleList([
            EncoderLayer(feature_dim, hidden_dim, n_head, d_k, d_v, att_type, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, input1, input2):
        enc_slf_attn_list1 = []
        enc_slf_attn_list2 = []


        for enc_layer in self.layer_stack:
            enc_output1, enc_slf_attn1 = enc_layer(input1, slf_attn_mask=None)
            enc_slf_attn_list1 += [enc_slf_attn1]

            enc_output2, enc_slf_attn2 = enc_layer(input2, slf_attn_mask=None)
            enc_slf_attn_list2 += [enc_slf_attn2]


        similarity = F.cosine_similarity(enc_output1, enc_output2, dim=1, eps=1e-8)
        return enc_output1, enc_output2, similarity, enc_slf_attn_list1, enc_slf_attn_list2

    # 归一化0-1
    def data_normal(self, origin_data):
        d_min = origin_data.min()
        if d_min < 0:
            origin_data += torch.abs(d_min)
            d_min = origin_data.min()
        d_max = origin_data.max()
        dst = d_max - d_min
        norm_data = (origin_data - d_min).true_divide(dst)
        return norm_data


# 损失函数定义
def MyLoss(pred, label):
    # criterion = nn.BCEWithLogitsLoss() # 二分类交叉熵
    criterion = nn.MSELoss(reduction='mean')  # 均方差
    loss = criterion(pred, label)
    return loss

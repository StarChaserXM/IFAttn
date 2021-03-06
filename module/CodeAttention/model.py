import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('.')
from module.CodeAttention.attention.SelfAttention import ScaledDotProductAttention
from module.CodeAttention.attention.ExternalAttention import ExternalAttention

class PositionwiseFeedForward(nn.Module):
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
        self.d_model = d_model
        self.slf_attn = ScaledDotProductAttention(d_model=self.d_model, d_k=d_k, d_v=d_v, h=n_head, dropout=dropout)
        self.ext_attn = ExternalAttention(d_model=self.d_model,S=d_k,dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        if self.att_type == 'SelfAttention':
            enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, attention_mask=slf_attn_mask)
        elif self.att_type == 'ExternalAttention':
            enc_output, enc_slf_attn = self.ext_attn(enc_input)
        elif self.att_type == 'NoAttention':
            enc_output = enc_input
            enc_slf_attn = None
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class SiameseAttentionNet(nn.Module):
    def __init__(self, feature_dim, hidden_dim, n_layers, n_head, d_k, d_v, att_type, dropout):
        super().__init__()
        self.feature_dim = feature_dim
        self.layer_stack = nn.ModuleList([
            EncoderLayer(feature_dim, hidden_dim, n_head, d_k, d_v, att_type, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, input1, input2):
        attn_list1 = []
        attn_list2 = []
        input1 = input1.unsqueeze(-1)
        input2 = input2.unsqueeze(-1)
        pad = (0,self.feature_dim-1)
        output1 = F.pad(input1,pad,'constant',0)
        output2 = F.pad(input2,pad,'constant',0)


        for enc_layer in self.layer_stack:
            output1, slf_attn1 = enc_layer(output1, slf_attn_mask=None)
            attn_list1 += [slf_attn1]

            output2, slf_attn2 = enc_layer(output2, slf_attn_mask=None)
            attn_list2 += [slf_attn2]


        output1 = output1.sum(dim=-1)
        output2 = output2.sum(dim=-1)
        similarity = F.cosine_similarity(output1, output2, dim=-1, eps=1e-8)
        return output1, output2, similarity, attn_list1, attn_list2

    def data_normal(self, origin_data):
        d_min = origin_data.min()
        if d_min < 0:
            origin_data += torch.abs(d_min)
            d_min = origin_data.min()
        d_max = origin_data.max()
        dst = d_max - d_min
        norm_data = (origin_data - d_min).true_divide(dst)
        return norm_data

def MyLoss(pred, label):
    criterion = nn.MSELoss(reduction='mean')
    loss = criterion(pred, label)
    return loss

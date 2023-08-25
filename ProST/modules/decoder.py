import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stop
from .Tran_utils import get_activation_fn
import pdb

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    


class Event_Layer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, is_weights=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn_vis = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.norm5 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True) # _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.is_weights = is_weights

    def forward(self, tgt, memory,
                     pos=None,
                     query_pos=None):

        tgt = self.norm1(tgt)
        memory = self.norm2(memory)
        tgt = self.self_attn(tgt, tgt, tgt)[0]
        tgt = self.norm3(tgt)
        #event
        tgt2, atten_weights = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm4(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm5(tgt)
        
        return tgt, atten_weights
        
#MSRVTT ada_para:0.5
def adaptive_mask(aa, bb, ada_para=0.5):
    tensor = torch.zeros((aa, bb))
    adaptive_num = int(bb * ada_para)
    cc = int(bb/aa)
    for i in range(aa):
        start_col = i* cc
        end_col = start_col + cc +  adaptive_num
        if end_col > bb-1:
            tmp = end_col - (bb-1)
            start_col = start_col - tmp
            if start_col < 0:
                start_col =0
            end_col = bb
        tensor[i, start_col:end_col] = 1
    tensor = ~tensor.bool()
    return tensor

class Frame_Layer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, para=1.0,  dropout=0.1,
                 activation="relu", normalize_before=False, is_weights=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn_vis = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU(inplace=True) 
        self.normalize_before = normalize_before
        self.is_weights = is_weights
        self.mask_para = para

    def forward(self, tgt, memory,
                     pos=None,
                     query_pos=None):
        tgt = self.norm1(tgt)
        memory = self.norm2(memory)
        mask_new = adaptive_mask(tgt.shape[0], memory.shape[0], ada_para = 0.2) 
        tgt2, atten_weights = self.multihead_attn(tgt, memory, memory, attn_mask=mask_new.cuda())
        tgt = tgt + self.dropout1(tgt2)

        tgt = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm4(tgt)
        
        return tgt, atten_weights
        

class TransDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                pos=None,
                query_pos=None):
        output = tgt

        intermediate = []
        all_weights = []

        for layer in self.layers:
            output, weights = layer(output, memory, 
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                all_weights.append(weights)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(all_weights)
        return output.unsqueeze(0)


class Event_decoder(nn.Module):
    def __init__(self, num_attris = 3, layers=1, heads=1, dim_ftr=512,
                 pos_emb=False, length=1, dim_feedforward=512, without_init=False):
        super().__init__()
        embedding_dim = dim_ftr

        d_model = dim_ftr
        dim_feedforward = dim_feedforward

        self.V = nn.Parameter(
            torch.Tensor(num_attris, dim_feedforward),
            requires_grad=True
        )
        nn.init.xavier_uniform_(self.V)
        decoder_layer = Event_Layer(d_model=d_model, nhead=heads, dim_feedforward=dim_feedforward)
        self.event_decoder = TransDecoder(decoder_layer, layers, nn.LayerNorm(d_model),
                                          return_intermediate=True)
        self.use_pos_enc = pos_emb

        if self.use_pos_enc:
            self.position_encoding_pre = positionalencoding2d(embedding_dim, 14, 14).unsqueeze(0)

    def forward(self, features):
        batch_size = features.shape[0]
        dim_num = features.shape[2] 
        if self.use_pos_enc: # False
            pos_encoding = self.position_encoding_pre(features,torch.zeros(features.shape[0],14,14, dtype=torch.bool).cuda())
            features = features + pos_encoding

        enco_others = features.permute(1, 0, 2)
        h_attr = self.V  

        query_seq_num = h_attr.shape[0]
        h_attr_batch = h_attr.unsqueeze(0).repeat(batch_size, 1, 1)
        h_attr_batch = h_attr_batch.permute(1, 0, 2)

        hs, _  = self.event_decoder(h_attr_batch, enco_others)
        hs = hs[-1].permute(1,0,2)
        return hs

class Frame_decoder(nn.Module):
    def __init__(self, num_attris = 3, layers=1, heads=1, dim_ftr=512,
                 pos_emb=False, length=1, dim_feedforward=512, without_init=False):
        super().__init__()
        embedding_dim = dim_ftr
        d_model = dim_ftr
        dim_feedforward = dim_feedforward

        self.V = nn.Parameter(
            torch.Tensor(num_attris, dim_feedforward),
            requires_grad=True
        )
        nn.init.xavier_uniform_(self.V)
        decoder_layer = Frame_Layer(d_model=d_model, nhead=heads, dim_feedforward=dim_feedforward)
        self.event_decoder = TransDecoder(decoder_layer, layers, nn.LayerNorm(d_model),
                                          return_intermediate=True)
        self.use_pos_enc = pos_emb

        if self.use_pos_enc:
            self.position_encoding_pre = positionalencoding2d(embedding_dim, 14, 14).unsqueeze(0)

    def forward(self, features):
        batch_size = features.shape[0]
        dim_num = features.shape[2] 
        if self.use_pos_enc:  
            pos_encoding = self.position_encoding_pre(features,torch.zeros(features.shape[0],14,14, dtype=torch.bool).cuda())
            features = features + pos_encoding

        enco_others = features.permute(1, 0, 2) 
        h_attr = self.V  

        query_seq_num = h_attr.shape[0]
        h_attr_batch = h_attr.unsqueeze(0).repeat(batch_size, 1, 1)
        h_attr_batch = h_attr_batch.permute(1, 0, 2)

        hs, _  = self.event_decoder(h_attr_batch, enco_others)
        hs = hs[-1].permute(1,0,2)

        return hs
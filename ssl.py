
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 23:56:05 2025

@author: sami_b
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 12:46:53 2025

@author: sami_b
"""
import itertools
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts
from real_data import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
# Cell
from typing import Callable, Optional
from torch import Tensor
import torch.nn.functional as F
import math
import random
import numpy as np
from torch.utils.data import DataLoader

import os

import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, random_split, TensorDataset


seed=2025


def set_seed(seed=2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    torch.backends.cudnn.deterministic = True  # slower but deterministic
    torch.backends.cudnn.benchmark = False     # disables autotuner that can introduce randomness
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    print(f"Random seed set to {seed}")
    
    

def seed_worker(worker_id):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
set_seed(seed)   
g = torch.Generator()
g.manual_seed(seed)


#our proposed contiguous masking method
def create_mask_with_contig_block(bs, num_patch, mask_ratio,max_n_contig): #24hours
    mask = torch.rand(bs, num_patch) < mask_ratio # mask_ratio is actually 1-true_ratio

    for i in range(bs):
        n_contig = random.randint(1, max_n_contig)
        if num_patch - n_contig <= 0:
            start_idx = 0
        else:
            start_idx = random.randint(0, num_patch - n_contig)

        mask[i, start_idx:start_idx + n_contig] = True  # Force masking of contiguous block

    return mask


#patchtst authors' masking
def apply_patch_masking(x, mask_ratio=0.3):#apply_patch_masking(x, mask_ratio=0.3)[0] to extract X_MASKED
    """
    Mask a fraction of patches in tensor `x` of shape [B, C, patch_len, patch_num].
    Returns masked input and a binary mask of which patches were masked.
    for masking ratio = 0.3 and num_patches=6, it masks a single patch
    """
    B, C, Lp, P = x.shape
    num_mask = int(P * mask_ratio)

    masks = torch.zeros(B, P, device=x.device)
    for i in range(B):
        idx = torch.randperm(P)[:num_mask]
        masks[i, idx] = 1

    mask_expand = masks[:, None, None, :].expand(-1, C, Lp, -1)
    x_masked = x.clone()
    x_masked[mask_expand == 1] = 0

    return x_masked, masks



#instance normalization, from patchtst code
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

    
    

    
# pos_encoding, from patchtst code
def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe

SinCosPosEncoding = PositionalEncoding

def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    x = .5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        pv(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)
        if abs(cpe.mean()) <= eps: break
        elif cpe.mean() > eps: x += .001
        else: x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1)**(.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe

def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d': W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d': W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos': W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)
    
  
    
#the different heads are from patchtst code
class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x



class PredictionHead(nn.Module):
    def __init__(self, individual, n_vars, d_model, num_patch, forecast_len, head_dropout=0, flatten=False):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars
        self.flatten = flatten
        head_dim = d_model*num_patch


        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(head_dim, forecast_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
        
            self.linear = nn.Linear(head_dim, forecast_len)
            self.dropout = nn.Dropout(head_dropout)



    def forward(self, x):                     
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x forecast_len x nvars]
        """
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * num_patch]
                z = self.linears[i](z)                    # z: [bs x forecast_len]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)         # x: [bs x nvars x forecast_len]
        else:
            
            x = self.flatten(x)     # x: [bs x nvars x (d_model * num_patch)]    
            x = self.dropout(x)

            x = self.linear(x)      # x: [bs x nvars x forecast_len]

        return x.transpose(2,1)     # [bs x forecast_len x nvars]
    


class PretrainHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        """
        x: tensor [bs x nvars x d_model x num_patch]
        output: tensor [bs x nvars x num_patch x patch_len]
        """

        x = x.transpose(2,3)                     # [bs x nvars x num_patch x d_model]
        x = self.linear( self.dropout(x) )      # [bs x nvars x num_patch x patch_len]
        x = x.permute(0,2,1,3)                  # [bs x num_patch x nvars x patch_len]
        return x



# tstencoder class from patchtst code
class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False,talking_heads=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn,talking_heads=talking_heads) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor):
        """
        src: tensor [bs x q_len x d_model]
        """
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores)
            return output
        else:
            for mod in self.layers: output = mod(output)
            return output



#tstencoderlayer is basically from patchtst code. We have only added glu as another choice for activation function in the ffn
class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, 
                activation="gelu", res_attention=False, pre_norm=False,talking_heads=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention,talking_heads=talking_heads)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        if activation=="glu":
            self.ff = nn.Sequential(nn.Linear(d_model, 2*d_ff, bias=bias),
                        get_activation_fn(activation),
                        nn.Dropout(dropout),
                        nn.Linear(d_ff, d_model, bias=bias))
        else:
            self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(self, src:Tensor, prev:Optional[Tensor]=None):
        """
        src: tensor [bs x q_len x d_model]
        """
        # Multi-Head attention sublayer
        if self.pre_norm: #by default false. if true, batch norm if batch or layer norm if not
            src = self.norm_attn(src)
        
            # Q:       [batch_size (bs) x max_q_len x d_model]
            # K, V:    [batch_size (bs) x q_len x d_model]
            # src,src,src refer to q,k,v

        ## Multi-Head attention
        if self.res_attention: #by default false
            src2, attn, scores = self.self_attn(src, src, src, prev)
        else:
            src2, attn = self.self_attn(src, src, src)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
     
        if not self.pre_norm:
            src = self.norm_attn(src)
        

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        
        if not self.pre_norm:
            src = self.norm_ffn(src)



        if self.res_attention:
            return src, scores
        else:
            return src




#our modified multiheadattention module includes the addition of hybrid norm, rope, and talking-heads mechanism
class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False,talking_heads=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        
        #for QKV normalization: 
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa,talking_heads=talking_heads)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q
        
        # #for QKV normalization
        Q = self.norm_q(Q)
        K = self.norm_k(K)
        V = self.norm_v(V)
        
        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k] = (16,8,23,128/8=16)
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


##################### rotary positional embedding#############
        ##step1" Define RoPE helper function
def apply_rotary_pos_emb(x, sin, cos):
    # x: [batch, heads, seq_len, head_dim]
    x1, x2 = x[..., ::2], x[..., 1::2]
    x_rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1) #eq34 of the paper "ROFORMER: ENHANCED TRANSFORMER WITH ROTARY POSITION EMBEDDING"
    return x_rotated

    #step2: Generate sinusoidal angles (once per forward pass)
    
def get_sin_cos(seq_len, head_dim, device):
    position = torch.arange(seq_len, device=device).unsqueeze(1)
    freq = torch.pow(10000, -torch.arange(0, head_dim, 2, device=device) / head_dim)
    angles = position * freq
    return torch.sin(angles), torch.cos(angles)

###################################

class ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False,talking_heads=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa
        self.talking_heads = talking_heads
        if talking_heads:
            self.pre_softmax_proj = nn.Linear(n_heads, n_heads, bias=False)
            self.post_softmax_proj = nn.Linear(n_heads, n_heads, bias=False)

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''
        #option 1: standard/absolute attention
        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        #attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]
       
        #option2: rope
       #  #q, k: [bs, heads, seq_len, head_dim]
        seq_len = q.size(-2)
        head_dim = q.size(-1)
        sin, cos = get_sin_cos(seq_len, head_dim, q.device)  # [seq_len, head_dim/2]
        # reshape sin, cos for broadcasting
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim/2]
        cos = cos.unsqueeze(0).unsqueeze(0)
        # Apply rotary embedding
        q = apply_rotary_pos_emb(q, sin, cos)
        k = apply_rotary_pos_emb(k.transpose(-2, -1), sin, cos).transpose(-2, -1)
       # Compute scaled dot-product attention
        attn_scores = torch.matmul(q, k) * self.scale

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev
        if self.talking_heads:
            attn_scores = attn_scores.permute(0, 2, 3, 1)  # [bs, q_len, k_len, heads]
            attn_scores = self.pre_softmax_proj(attn_scores)
            attn_scores = attn_scores.permute(0, 3, 1, 2)  # [bs, heads, q_len, k_len]

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)
        
        if self.talking_heads:
            attn_weights = attn_weights.permute(0, 2, 3, 1)  # [bs, q_len, k_len, heads]
            attn_weights = self.post_softmax_proj(attn_weights)
            attn_weights = attn_weights.permute(0, 3, 1, 2)  # [bs, heads, q_len, k_len]

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):        
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)



def get_activation_fn(activation):
    if callable(activation): return activation()
    elif activation.lower() == "relu": return nn.ReLU()
    elif activation.lower() == "gelu": return nn.GELU()
    elif activation.lower() == "glu": return nn.GLU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')
    
    

#we tested the combination of lstmencoder with patchtst encoder, but this did not lead to better results. Lstmencoder does not have any role in the code    
class LSTMEncoder(nn.Module):  # renamed to better reflect purpose
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)

        # Optional: project hidden_size → input_size (d_model) if they're different
        self.proj = nn.Identity()
        if hidden_size != input_size:
            self.proj = nn.Linear(hidden_size, input_size)#nn.Sequential(
                # (nn.Linear(hidden_size, input_size)),
                #  nn.GELU(),
                #  )

    def forward(self, x):
        # x: [bs * n_vars, num_patch, d_model]
        out, _ = self.lstm(x)  # [bs * n_vars, num_patch, hidden_size]
        out = self.dropout(out)
        out = self.proj(out)  # [bs * n_vars, num_patch, d_model]
        return out
    
    
class PatchTSTEncoder(nn.Module):
    def __init__(self, c_in, num_patch, patch_len, hidden_lstm,layer_lstm,dropout_lstm,
                 n_layers=3, d_model=128, n_heads=16, shared_embedding=True,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False,talking_heads=False, **kwargs):

        super().__init__()
        self.n_vars = c_in
        self.num_patch = num_patch #-1 because i added one toi account for overlapping
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding        

        # Input encoding: projection of feature vectors onto a d-dim vector space
        if not shared_embedding: 
            self.W_P = nn.ModuleList()
            for _ in range(self.n_vars): self.W_P.append(nn.Linear(patch_len, d_model))
        else:
            self.W_P = nn.Linear(patch_len, d_model)      #learning of commom features between variables

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, num_patch, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, 
                                    store_attn=store_attn,talking_heads=talking_heads)
        self.fc=nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )


        self.lstm=LSTMEncoder(input_size=d_model, hidden_size=hidden_lstm, num_layers=layer_lstm, dropout=dropout_lstm) 

    def forward(self, x) -> Tensor:          
        """
        x: tensor [bs x num_patch x nvars x patch_len]
        """
        bs, num_patch, n_vars, patch_len = x.shape
        # Input encoding
        if not self.shared_embedding:
            x_out = []
            for i in range(n_vars): 
                z = self.W_P[i](x[:,:,i,:])
                x_out.append(z)
            x = torch.stack(x_out, dim=2)
        else: #captures the inter-variable correlations implicitly
            x = self.W_P(x)                                                      # x: [bs x num_patch x nvars x d_model]
        x = x.transpose(1,2)                                                     # x: [bs x nvars x num_patch x d_model]        

        u = torch.reshape(x, (bs*n_vars, num_patch, self.d_model) )    
        ###### uncomment if you want to use absolute pe
        # if self.W_pos.size(1) != u.size(1):
        #     # Interpolate W_pos to match new number of patches
        #     W_pos = self.W_pos.unsqueeze(0).permute(0, 2, 1)  # [1, d_model, 24]
        #     W_pos = F.interpolate(W_pos, size=u.size(1), mode='linear', align_corners=False)  # → [1, d_model, 47]
        #     W_pos = W_pos.permute(0, 2, 1)  # [1, 47, d_model]
        # else:
        #     W_pos = self.W_pos          # u: [bs * nvars x num_patch x d_model]
        u = self.dropout(u)# + W_pos)#remove + w_pos if you want to use rotary pe self from w_pos                                         # u: [bs * nvars x num_patch x d_model]

        # Encoder
     
        z = self.encoder(u)                                                      # z: [bs * nvars x num_patch x d_model]
        #l=self.lstm(u)
        #z=z+l#+ cnn_out#self.alpha*(z)+(1-self.alpha)*l#+u
        #without cross-channel attention
        z = torch.reshape(z, (-1,n_vars, num_patch, self.d_model))               # z: [bs x nvars x num_patch x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x num_patch]
        



        return z

    
#our mofidied patch class
class PatchTST(nn.Module):
    """
    Output dimension: 
         [bs x target_dim x nvars] for prediction
         [bs x target_dim] for regression
         [bs x target_dim] for classification
         [bs x num_patch x n_vars x patch_len] for pretrain
    """
    def __init__(self, c_in:int, target_dim:int, patch_len:int, stride:int, num_patch:int, 
                 hidden_lstm:int,layer_lstm:int,dropout_lstm:int,
                 n_layers:int=3, d_model=128, n_heads=16, shared_embedding=True, d_ff:int=256, 
                 norm:str='BatchNorm', attn_dropout:float=0.2, dropout:float=0., act:str="glu",  #attn_dropout from 0 to 0.2 #glu instead of gelu
                 res_attention:bool=True, pre_norm:bool=True, store_attn:bool=False, #pre_norm from False to True
                 pe:str='exp1d', learn_pe:bool=True, head_dropout = 0,  #pe from zeros to exp1d
                 head_type = "prediction", individual = False, revin=True,affine = True, subtract_last = False,
                 y_range:Optional[tuple]=None, verbose:bool=False,talking_heads=True, **kwargs):

        super().__init__()

        assert head_type in ['pretrain', 'prediction', 'regression', 'classification'], 'head type should be either pretrain, prediction, or regression'
        # Backbone
        #gating
        self.backbone = PatchTSTEncoder(c_in, num_patch=num_patch, patch_len=patch_len, 
                                hidden_lstm=hidden_lstm,layer_lstm=layer_lstm,dropout_lstm=dropout_lstm,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, 
                                shared_embedding=shared_embedding, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, 
                                res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose,talking_heads=talking_heads, **kwargs)

        # Head
        self.n_vars = c_in
        self.head_type = head_type

        if head_type == "pretrain":
            self.head = PretrainHead(d_model, patch_len, head_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == "prediction":
            self.head = PredictionHead(individual, self.n_vars, d_model, num_patch, target_dim, head_dropout)
        elif head_type == "regression":
            self.head = RegressionHead(self.n_vars, d_model, target_dim, head_dropout, y_range)
        elif head_type == "classification":
            self.head = ClassificationHead(self.n_vars, d_model, target_dim, head_dropout)

        self.revin = revin
        #gating
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        self.stride=stride
        self.patch_len=patch_len

        self.d_model=d_model
        
    def forward(self, z):                             
        """
        z: tensor [bs x num_patch x n_vars x patch_len]
        """   

        # Merge patches if needed: [bs, num_patch, n_vars, patch_len] → [bs, seq_len, n_vars]
        if z.dim()>3:#pretraining: the samples are patched 
            z = z.permute(0, 2, 1, 3).contiguous()  # [bs, n_vars, num_patch, patch_len]
            z = z.view(z.size(0), z.size(1), -1).permute(0, 2, 1).contiguous()  # [bs, seq_len, n_vars]
        
        if self.revin:
            z = self.revin_layer(z, 'norm')  # Normalize per-sample
        # Restore patch structure
        #add a line that adds padding if the model is to be finetuned
        z = z.permute(0, 2, 1)  # [bs, n_vars, seq_len]

        z = z.unfold(dimension=2, size=self.patch_len, step=self.stride)
        z = z.permute(0, 2, 1, 3)  # [bs, num_patch, n_vars, patch_len] (num_patch+1 if head_type=="prediction")

        
        z = self.backbone(z)#make two cases in backbone, whether for pretrain or finetuning                                                                # z: [bs x nvars x d_model x num_patch]

        z = self.head(z)
       
       
        if self.revin and self.head_type == "prediction":

            
            z = self.revin_layer(z, 'denorm')  # Denormalize output
            #gating
            #z=z[:,:,:-2]
            
            #z = z.permute(0, 2, 1)  # [bs, target_len, nvars]                                                                    
        # z: [bs x target_dim x nvars] for prediction
        #    [bs x target_dim] for regression
        #    [bs x target_dim] for classification
        #    [bs x num_patch x n_vars x patch_len] for pretrain
        return z  



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import time
start_time = time.time()


####################################reproducibility of the results#####################################
#change the hyperparameters' values to the ones mentioned in the paper, depending on your dataset and forecasting horizon
param_grid = {
        "seq_len": [96],
        "patch_len": [16], #
        "d_model": [128],
        "n_heads": [16],
        "n_layers": [2],
        "d_ff": [128], #
        "dropout": [0.33],
        "lr": [1e-4], #
        "batch_size": [64],
        "mask_ratio": [0.4], 
        "weight_decay": [0.05],
        "K": [32], 
        "attn_dropout": [0.4],
        "act": ["glu"],
        "pre_norm": [True],
        "hidden_lstm": [128],
        "layer_lstm": [3],
        "dropout_lstm": [0.4]
}

                 
overlap_ratio=3
PT_EPOCHS =1
FT_EPOCHS=1
PRED_LEN = 336
dn='etth2'

###################################################################################################################
keys, values = zip(*param_grid.items())
all_combinations = list(itertools.product(*values))
num_samples = len(all_combinations)
sampled_combinations = random.sample(all_combinations, num_samples)


best_mse = float("inf")
best_mae = float("inf")
best_config = None
best_model_state = None


    # 2. Loop over configs
for count, config_values in enumerate(sampled_combinations, start=1):
    set_seed(seed)
    torch.cuda.empty_cache()
    config = dict(zip(keys, config_values))
    print(f"\n******** Grid {count} ********")
    print(f"Config: {config}")


    class Params:
        dset= dn#electricity
        context_points= config["seq_len"] #L
        target_points= PRED_LEN #T
        batch_size= config["batch_size"]
        num_workers= 8
        with_ray= False
        features='M'
        #use_time_features = True


    params = Params 
    dls = get_dls(params)
    
    from torch.utils.data import ConcatDataset, Subset
    
    # 1. Combine all original datasets
    full_dataset = ConcatDataset([dls.train.dataset, dls.valid.dataset, dls.test.dataset])
    total_len = len(full_dataset)
    indices = list(range(total_len))
    
    # 3. Split into 10 equal partitions
    part_size = total_len // 10
    splits = [indices[i*part_size : (i+1)*part_size] for i in range(10)]
    
    # 4. Recombine for each desired split
    train_pre_idx   = splits[0] + splits[1] + splits[2] + splits[3] + splits[4]
    val_pre_idx     = splits[5]
    train_ft_idx    = splits[6]
    val_ft_idx      = splits[7]
    final_train_idx = splits[6] + splits[7]
    test_ft_idx     = splits[8] + splits[9]
    
    # 5. Create Subsets
    train_pre_ds   = Subset(full_dataset, train_pre_idx)
    val_pre_ds     = Subset(full_dataset, val_pre_idx)
    train_ft_ds    = Subset(full_dataset, train_ft_idx)
    val_ft_ds      = Subset(full_dataset, val_ft_idx)
    final_train_ds = Subset(full_dataset, final_train_idx)
    test_ft_ds     = Subset(full_dataset, test_ft_idx)

    
    num_workers=8
    ##
    train_pre_loader = DataLoader(train_pre_ds, batch_size=config["batch_size"], shuffle=False, pin_memory=True,num_workers=num_workers,persistent_workers=True, worker_init_fn=seed_worker,generator=g)#remove sampler
    val_pre_loader   = DataLoader(val_pre_ds, batch_size=config["batch_size"], shuffle=False, pin_memory=True,num_workers=num_workers,persistent_workers=True, worker_init_fn=seed_worker,generator=g)
    train_ft_loader  = DataLoader(train_ft_ds, batch_size=config["batch_size"], shuffle=False, pin_memory=True,num_workers=num_workers,persistent_workers=True, worker_init_fn=seed_worker,generator=g)
    val_ft_loader    = DataLoader(val_ft_ds, batch_size=config["batch_size"], shuffle=False, pin_memory=True,num_workers=num_workers,persistent_workers=True, worker_init_fn=seed_worker,generator=g)
    final_train_loader=DataLoader(final_train_ds, batch_size=config["batch_size"], shuffle=False, pin_memory=True,num_workers=num_workers,persistent_workers=True, worker_init_fn=seed_worker,generator=g)
    test_ft_loader   = DataLoader(test_ft_ds, batch_size=config["batch_size"], shuffle=False, pin_memory=True,num_workers=num_workers,persistent_workers=True, worker_init_fn=seed_worker,generator=g)
    
    
    for x,y  in tqdm(train_pre_loader): #dls.train ~ train_pre_loader, dls.valid ~ valid_pre_loader
        x = x.to(DEVICE)
        break
    
    c_in=x.shape[-1]
    STRIDE= config["patch_len"]

    #parallel
    model = PatchTST(
        c_in=c_in,
        n_layers=config["n_layers"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        attn_dropout=config["attn_dropout"],
        act=config["act"],
        pre_norm=config["pre_norm"],
        target_dim=None,
        patch_len=config["patch_len"],
        stride=STRIDE,
        num_patch=(config["seq_len"] - config["patch_len"]) // STRIDE + 1,
        hidden_lstm=config['hidden_lstm'],
        layer_lstm=config['layer_lstm'],
        dropout_lstm=config['dropout_lstm'],
        head_type='pretrain',
    ).to(DEVICE)


    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],weight_decay=config["weight_decay"])
    eta_min = config["lr"]/10      # Final learning rate
    # 3. Create the scheduler
    T_max = PT_EPOCHS#//10  # Total number of epochs for cosine cycle 10% of the total number of epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_max, eta_min=eta_min)
 
    best_val_loss = float('inf')
    patience = 10  # Number of epochs to wait
    epochs_no_improve = 0


    
    train_losses = []
    val_losses = []

    for epoch in range(PT_EPOCHS):

        model.train()
        train_loss = 0.0
        for x,_ in tqdm(train_pre_loader):

            x = x.to(DEVICE)

            x = x.squeeze(-1)
            
            #patchifying
            # x: [bs, seq_len, n_vars] → permute to [bs, n_vars, seq_len]
            x = x.permute(0, 2, 1)
            # unfold to get sliding windows → [bs, n_vars, num_patches, patch_len]
            x = x.unfold(dimension=2, size=config["patch_len"], step=STRIDE)
            # permute to [bs, num_patches, n_vars, patch_len]
            x = x.permute(0, 2, 1, 3) 

            patches=x
            
            
            mask_ratio = config["mask_ratio"]  #the masking ratio is per batch, it could be for example that a sample is not masked at all. masking per 
            #batch offers more masking diversity. PS :  i can make an ablation study later by comparing the results by masking per batch vs
            # masking per sample (in terms of masking ratio). also, i can do an ablation study on the masking values: masking with zeros vs 
            # masking with random values
            # x: [bs, num_patch, n_vars, patch_len]
            bs, num_patch, n_vars, patch_len = x.shape
            num_mask = int(num_patch * mask_ratio)
            # # option1: contiguous mask
            mask=create_mask_with_contig_block(bs, num_patch, mask_ratio,max_n_contig=config["K"]//patch_len)
            mask = mask.to(x.device)
            x_masked = x.clone()
            x_masked[mask.unsqueeze(-1).unsqueeze(-1).expand_as(x)] = 0  # Zero-out masked patches
            masked_patches=x_masked
            # #option2: authors' mask
            #mask=apply_patch_masking(x, mask_ratio=mask_ratio)[1].int()
            # masked_patches=apply_patch_masking(x, mask_ratio=mask_ratio)[0]
            # 3. Forward pass
            outputs = model(masked_patches)  # [bs, num_patch, n_vars, patch_len]
            
            # 4. Compute loss only on masked patches
            #option1:
            loss=F.mse_loss(outputs[mask], patches[mask])

            
            # 5. Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_pre_loader)

        scheduler.step()

        print(f"Epoch {epoch+1}, LR: {scheduler.get_last_lr()[0]:.6f}")

        # ----- Validation -----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _ in tqdm(val_pre_loader, desc=f"Epoch {epoch+1} [Val]"):
                #x = torch.cat([x, x_calendar[:,:,-2:]], dim=-1)
                x = x.to(DEVICE).squeeze(-1)
                x = x.permute(0, 2, 1).unfold(dimension=2, size=config["patch_len"], step=STRIDE)
                x = x.permute(0, 2, 1, 3)
                patches = x#[:,:,:-2]
    
                bs, num_patch, n_vars, patch_len = x.shape
                #option 1: contiguous masking
                mask=create_mask_with_contig_block(bs, num_patch, mask_ratio,max_n_contig=config["K"]//patch_len)
                mask = mask.to(x.device)
                x_masked = x.clone()
                x_masked[mask.unsqueeze(-1).unsqueeze(-1).expand_as(x)] = 0
                #option2: authors' mask
                # mask=apply_patch_masking(x, mask_ratio=mask_ratio)[1].int()
                # x_masked=apply_patch_masking(x, mask_ratio=mask_ratio)[0]
    
                outputs = model(x_masked)
                loss = F.mse_loss(outputs[mask], patches[mask])

                val_loss += loss.item()
    
        avg_val_loss = val_loss / len(val_pre_loader)
        if avg_val_loss<best_val_loss:
            best_val_loss=avg_val_loss
            epochs_no_improve = 0

            torch.save(model.state_dict(), f"patchtst_pretrained_best_{PRED_LEN}_{dn}.pth")

            print(f"✅ Best model saved with val loss: {best_val_loss:.6f}")
        print(f"Epoch {epoch+1}/{PT_EPOCHS} → Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        else:
            epochs_no_improve += 1
            if epochs_no_improve > patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

    import matplotlib.pyplot as plt

    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='s', color='orange')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('pretraining Training vs. Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Load encoder weights
    pretrained = PatchTST(
        c_in=c_in,
        n_layers=config["n_layers"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        attn_dropout=config["attn_dropout"],
        act=config["act"],
        pre_norm=config["pre_norm"],
        target_dim=None,
        patch_len=config["patch_len"],
        stride=STRIDE,
        num_patch=(config["seq_len"] - config["patch_len"]) // STRIDE + 1,
        hidden_lstm=config['hidden_lstm'],
        layer_lstm=config['layer_lstm'],
        dropout_lstm=config['dropout_lstm'],
        head_type='pretrain',
    ).to(DEVICE)
    
    pretrained.load_state_dict(torch.load(f"patchtst_pretrained_best_{PRED_LEN}_{dn}.pth")) 

 
    # Build forecasting model
    #the patches are chosen to be overlapping because this increases the receptive field and provides finer-grained temporal resolution

    forecast_model = PatchTST(
        c_in=c_in,
        target_dim=PRED_LEN,
        patch_len=config["patch_len"],
        n_layers=config["n_layers"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        attn_dropout=config["attn_dropout"],
        act=config["act"],
        pre_norm=config["pre_norm"],
        stride=STRIDE//overlap_ratio,
        num_patch=(config["seq_len"] - config["patch_len"]) // (STRIDE//overlap_ratio) + 1,#+2 to account for overlapping patches
        hidden_lstm=config['hidden_lstm'],
        layer_lstm=config['layer_lstm'],
        dropout_lstm=config['dropout_lstm'],
        head_type='prediction',
        individual=False
    ).to(DEVICE)

    
    # Only replace head weights? (Optional: freeze encoder)
    forecast_model.backbone = pretrained.backbone
    


    forecast_model.head = PredictionHead(
        individual=False,  # or False depending on your setting
        #d_model=d_model,
        num_patch=(config["seq_len"] - config["patch_len"]) // (STRIDE//overlap_ratio) + 1,  # +2 for overlapping patches
        n_vars=c_in,
        d_model=config["d_model"],  
        forecast_len=PRED_LEN
    ).to(DEVICE)
    

    
    
    T_max = FT_EPOCHS#//3 
    optimizer = torch.optim.AdamW(forecast_model.parameters(), lr=config["lr"],weight_decay=config["weight_decay"])
    criterion = nn.MSELoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=config["lr"]/10)
    #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_max, eta_min=eta_min) 
    
    best_val_loss = float('inf')
    best_ft_model_path = f'best_ft_model_{PRED_LEN}_{dn}.pth'
    patience = 7  # Number of epochs to wait
    epochs_no_improve = 0
    
    

    
    #option 3: linear probing and fine tune the head
    # Freeze encoder parameters
    for param in forecast_model.backbone.parameters():
        param.requires_grad = False
    
    train_losses = []
    val_losses = []
    for epoch in range(FT_EPOCHS):
        #parallel
       # train_ft_sampler.set_epoch(epoch)
        forecast_model.train()
        #lstm.train()
        total_loss = 0
        for x, y in train_ft_loader:
            #parallel
            #x = torch.cat([x, x_calendar[:,:,-2:]], dim=-1)
            x = x.squeeze(-1).to(DEVICE)#.unsqueeze(2)
            y = y.squeeze(-1).to(DEVICE)  # [B, PRED_LEN]

            out = forecast_model(x)  # [B, PRED_LEN, 1]
            out = out.squeeze(-1)    # [B, PRED_LEN]

            
            loss = criterion(out, y.squeeze(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(forecast_model.parameters(),max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        print(f"Forecast epoch {epoch+1}, loss = {total_loss / len(train_ft_loader):.4f}")

        avg_train_loss = total_loss / len(train_ft_loader)
        scheduler.step()
        
        print(f"Epoch {epoch+1}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        forecast_model.eval()

        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_ft_loader:
                x_val, y_val = x_val.squeeze(-1).to(DEVICE), y_val.squeeze(-1).to(DEVICE)

                out = forecast_model(x_val)  # [B, PRED_LEN, 1]
                out = out.squeeze(-1)

                val_batch_loss = criterion(out, y_val.squeeze(-1))
                val_loss += val_batch_loss.item()
        
            avg_val_loss = val_loss / len(val_ft_loader)
            print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.6f}")
        
            # Save model if validation loss improves
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save(forecast_model.state_dict(), best_ft_model_path)
                #torch.save(lstm.state_dict(), "best_lstm.pth")
                print(f"✅ Saved new best model with val loss {best_val_loss:.6f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve > patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
    ##
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='s', color='orange')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('finetuning Training vs. Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

 

    
   ################################################################# retrain using train+val

    forecast_model.load_state_dict(torch.load(best_ft_model_path)) 
    train_losses = []
    for epoch in range(FT_EPOCHS):

        forecast_model.train()
        total_loss = 0
        for x, y in final_train_loader:

            x = x.squeeze(-1).to(DEVICE)#.unsqueeze(2)
            y = y.squeeze(-1).to(DEVICE)  # [B, PRED_LEN]

            out = forecast_model(x)  # [B, PRED_LEN, 1]
            out = out.squeeze(-1)    # [B, PRED_LEN]
            loss = criterion(out, y.squeeze(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(forecast_model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        print(f"Final training epoch {epoch+1}, loss = {total_loss / len(final_train_loader):.4f}")
        avg_train_loss = total_loss / len(final_train_loader)
        scheduler.step()
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    torch.save(forecast_model.state_dict(), best_ft_model_path)
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='final Training Loss', marker='o', color='blue')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('final training loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
 
    
 ################################## testing
    forecast_model.load_state_dict(torch.load(best_ft_model_path)) 
    forecast_model.eval()
    #lstm.load_state_dict(torch.load("best_lstm.pth")) 
    #lstm.eval()
    preds, trues, inputs = [], [], []
    with torch.no_grad():
        for x, y in test_ft_loader:
            #x = torch.cat([x, x_calendar[:,:,-2:]], dim=-1)
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.squeeze(-1).to(DEVICE)#.unsqueeze(2)
            y = y.squeeze(-1).to(DEVICE)  # [B, PRED_LEN]

            out = forecast_model(x)  # [B, PRED_LEN, 1]
            out = out.squeeze(-1)    # [B, PRED_LEN]
            
            preds.append(out.cpu().numpy())
            trues.append(y.squeeze(-1).cpu().numpy())                
            inputs.append(x.squeeze(-1).squeeze(-1).cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    inputs = np.concatenate(inputs, axis=0) 

    from copy import deepcopy
    mse = np.mean((preds - trues) ** 2)
    mae = np.mean(np.abs(preds - trues))
    print(f"Test MSE: {mse:.6f}, MAE: {mae:.6f}")
    if mse < best_mse:
        best_mse = mse
        best_mae = mae
        best_config = deepcopy(config)
        best_model_state = deepcopy(forecast_model.state_dict())
        # Save best model
        torch.save(best_model_state, f"final_model_{PRED_LEN}_{dn}.pth")

        # File name includes prediction length
        filename = f"best_config_predlen_{PRED_LEN}_{dn}.txt"
        
        # Save to file
        with open(filename, "w") as f:
            f.write(f"Best Config (PRED_LEN={PRED_LEN}):\n")
            for k, v in best_config.items():
                f.write(f"{k}: {v}\n")
            f.write(f"\nTest MSE: {best_mse:.6f}\n")
            f.write(f"Test MAE: {best_mae:.6f}\n")
        
        print(f"Best configuration saved to '{filename}'")
        sample_idx = 20
        input_len = inputs.shape[1]
        pred_len = preds.shape[1]
        
        # Time axes
        t_input = np.arange(input_len)
        t_pred = np.arange(input_len, input_len + pred_len)
        
        
        #generate a random integer between 0 and c_in-1
        import random
        dimension=random.randint(0, c_in - 1)
        
        data = {
        'Time': t_pred,  # Time values
        'True': trues[sample_idx][:, dimension],  # True values
        'Prediction': preds[sample_idx][:, dimension]  # Predicted values
        }
        
        # Convert data to DataFrame
        df = pd.DataFrame(data)
        
        # Export to CSV
        df.to_csv(f'{dn}_PRED_LEN_{PRED_LEN}_true_vs_pred.csv', index=False)
        
        plt.figure()
        
        # Input history
        plt.plot(t_input, inputs[sample_idx][:,dimension], label='Input', color='blue')
        
        # Ground truth
        plt.plot(t_pred, trues[sample_idx][:,dimension], label='True', color='green')
        
        # Prediction
        plt.plot(t_pred, preds[sample_idx][:,dimension], label='Prediction', color='red', linestyle='--')
        
        #plt.title("PatchTST Forecast with Input History")
        plt.xlabel("time")
        plt.ylabel("value")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        # Save the figure BEFORE showing it
        filename = f"{dn}_PRED_LEN_{PRED_LEN}.png"
        plt.savefig(filename)
        print(f"Figure saved as '{filename}'")
        plt.show()


    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")


    
    
 
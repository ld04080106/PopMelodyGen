#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model.py

Author:      <ludy>
Modified:    2026/06/10
Licence:     <MIT>
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTMelody(nn.Module):
    def __init__(self, vocab_size, 
                       d_model=512, 
                       n_head=8, 
                       n_layer=6, 
                       dropout=0.1, 
                       max_len=2048):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(enc_layer, num_layers=n_layer)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self.max_len = max_len

    def forward(self, x, attn_mask):
        """
        x: [B, T] token ids
        attn_mask: [B, T] bool, True=valid token
        """
        B, T = x.shape
        if T > self.max_len:
            raise ValueError(f"T={T} exceeds max_len={self.max_len}")

        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.tok_emb(x) + self.pos_emb(pos)
        h = self.drop(h)

        # causal mask: [T, T] with True meaning "masked"
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        # key_padding_mask: True means "PAD to ignore"
        key_padding = ~attn_mask  # [B,T]

        h = self.tr(h, mask=causal, src_key_padding_mask=key_padding)
        h = self.ln(h)
        logits = self.head(h)  # [B,T,V]
        return logits

def weighted_token_ce_loss(logits, targets, attn_mask, sample_weight):
    """
    logits: [B,T,V]
    targets: [B,T]
    attn_mask: [B,T] True=valid
    sample_weight: [B] float
    """
    B, T, V = logits.shape
    loss_tok = F.cross_entropy(
        logits.reshape(B*T, V),
        targets.reshape(B*T),
        reduction="none"
    ).reshape(B, T)

    # mask PAD
    loss_tok = loss_tok * attn_mask.float()

    # 每个样本平均 token loss
    denom = attn_mask.float().sum(dim=1).clamp_min(1.0)
    loss_per_sample = loss_tok.sum(dim=1) / denom  # [B]

    # 加权
    loss = (loss_per_sample * sample_weight).mean()
    return loss

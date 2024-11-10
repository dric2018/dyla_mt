import numpy as np

#torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary

from tqdm import tqdm

# lightning
import pytorch_lightning as pl

from config import Config
import dataset

import logging
logging.basicConfig(level='INFO')

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from rich.traceback import install 
install()

from utils import utils
import wandb

class FlashAttention(nn.Module):
    def __init__(
        self, 
        embed_dim:int=Config.EMBEDDING_DIM, 
        num_heads:int=Config.N_HEADS,
        dropout=Config.ATTN_DROP_RATE
    ):
        super(FlashAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.scaling = embed_dim ** -0.5

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(dim=2)

        # Flash attention: Calculate attention using scaled dot-product in a memory-efficient way
        attn_logits = (q @ k.transpose(-2, -1)) * self.scaling
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))
        attn_probs = F.softmax(attn_logits, dim=-1)
        attn_probs = F.dropout(attn_probs, self.dropout, self.training)

        attn_output = (attn_probs @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(attn_output)
    
class FeedForward(nn.Module):
    def __init__(
        self, 
        embed_dim:int, 
        hidden_dim:int=Config.D_FF, 
        dropout:float=Config.FF_DROP_RATE
    ):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

class TransformerLayer(nn.Module):
    def __init__(
        self, 
        embed_dim, 
        num_heads, 
        hidden_dim, 
        dropout=0.1
    ):
        super(TransformerLayer, self).__init__()
        self.self_attn = FlashAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.self_attn(self.norm1(x), mask=mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self, 
        num_layers, 
        embed_dim, 
        num_heads, 
        hidden_dim, 
        dropout=0.1
    ):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerLayer(embed_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

class TransformerDecoder(nn.Module):
    def __init__(
        self, 
        num_layers, 
        embed_dim, 
        num_heads, 
        hidden_dim, 
        dropout=0.1
    ):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([TransformerLayer(embed_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)])

    def forward(self, x, enc_output, mask=None, enc_mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
            # Add cross-attention to encoder output (optional for some implementations)
        return x

class ByT5Model(nn.Module):
    def __init__(
        self, 
        vocab_size:int=Config.VOCAB_SIZE, 
        embed_dim:int=Config.EMBEDDING_DIM, 
        num_heads:int=Config.N_HEADS, 
        hidden_dim:int=Config.D_FF, 
        num_encoder_layers:int=Config.NUM_ENCODER_LAYERS, 
        num_decoder_layers:int=Config.NUM_DECODER_LAYERS, 
        enc_dropout:float=Config.ENCODER_DROPOUT,
        dec_dropout:float=Config.DECODER_DROPOUT
    ):
        super(ByT5Model, self).__init__()
        self.encoder = TransformerEncoder(num_encoder_layers, embed_dim, num_heads, hidden_dim, enc_dropout)
        self.decoder = TransformerDecoder(num_decoder_layers, embed_dim, num_heads, hidden_dim, dec_dropout)
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None):
        # Encode
        x = self.embed_tokens(input_ids)
        enc_output = self.encoder(x, mask=attention_mask)

        # Decode
        y = self.embed_tokens(decoder_input_ids)
        dec_output = self.decoder(y, enc_output, mask=attention_mask)

        return self.lm_head(dec_output)

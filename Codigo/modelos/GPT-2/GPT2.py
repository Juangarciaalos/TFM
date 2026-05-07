import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math

class GPT2Attention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1, use_sdpa=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_sdpa = use_sdpa
        self.dropout_p = dropout
        
        # Proyección unificada para Query, Key, Value
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_sdpa:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            mask = torch.tril(
                torch.ones(seq_len, seq_len, device=x.device)
            ).view(1, 1, seq_len, seq_len)

            scores = scores.masked_fill(mask == 0, float('-inf'))
            
            attn = torch.nn.functional.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.out_proj(out)

class GPT2Block(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1, use_sdpa=False):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = GPT2Attention(hidden_dim, num_heads, dropout, use_sdpa=use_sdpa)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2(nn.Module):
    def __init__(self, vocab_size=50257, hidden_dim=768, num_layers=12, num_heads=12, max_seq_len=1024, use_sdpa=False):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, hidden_dim)
        self.wpe = nn.Embedding(max_seq_len, hidden_dim)
        
        self.blocks = nn.ModuleList([GPT2Block(hidden_dim, num_heads, use_sdpa=use_sdpa) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(hidden_dim)
        
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        
        self.use_checkpoint = False

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        x = self.wte(input_ids) + self.wpe(position_ids)
        
        for layer in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
                
        x = self.ln_f(x)
        logits = self.lm_head(x) 
        return logits

def gpt2_base(use_sdpa=False):
    return GPT2(hidden_dim=768, num_layers=12, num_heads=12, use_sdpa=use_sdpa)

def gpt2_medium(use_sdpa=False):
    return GPT2(hidden_dim=1024, num_layers=24, num_heads=16, use_sdpa=use_sdpa)

def gpt2_large(use_sdpa=False):
    return GPT2(hidden_dim=1280, num_layers=36, num_heads=20, use_sdpa=use_sdpa)
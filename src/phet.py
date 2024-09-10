# s/o Karpathy: https://github.com/karpathy/nanoGPT/blob/master/model.py

import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.nn import functional as F


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.h_dim % config.n_heads == 0
        self.config = config
        self.n_heads = config.n_heads
        self.h_dim = config.h_dim
        
        self.qkv_proj = nn.Linear(config.h_dim, 3 * config.h_dim, bias=False)
        self.o_proj = nn.Linear(config.h_dim, config.h_dim, bias=False)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    
    def forward(self, x, mask=None):
        B, L, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.qkv_proj(x).split(self.h_dim, dim=2)
        k = k.view(B, L, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, L, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, L, self.n_heads, C // self.n_heads).transpose(1, 2)
        
        # attention
        if self.flash:
            if mask is not None:
                mask = mask.unsqueeze(1).unsqueeze(2)
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask.bool(), dropout_p=self.config.dropout if self.training else 0, is_causal=False)
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.config.dropout if self.training else 0, is_causal=False)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if mask is not None:
                mask = mask.unsqueeze(1).unsqueeze(2)
                mask = mask.expand(B, self.n_heads, L, L)
                att = att.masked_fill(mask == 0, float('-inf'))
                
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, L, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.o_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.h_dim, 4*config.h_dim, bias=False)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4*config.h_dim, config.h_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.h_dim, eps=config.ln_eps)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.h_dim, eps=config.ln_eps)
        self.mlp = MLP(config)
        
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class PhETConfig:
    p_size = 25
    v_size = 665
    n_layers = 12
    n_heads = 12
    h_dim = 768
    ln_eps = 1e-12
    dropout = 0.1
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid config parameter: {key}")
    
    def to_dict(self):
        return {
            attr: getattr(self, attr)
            for attr in vars(self)
            if not attr.startswith('__') and not callable(getattr(self, attr))
        }


class PhET(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.ModuleDict(dict(
            value_embeds = nn.Embedding(config.v_size, config.h_dim),
            phenotype_embeds = nn.Embedding(config.p_size, config.h_dim),
        ))
        self.ln = nn.RMSNorm(config.h_dim, eps=config.ln_eps)
        self.encoder = nn.ModuleDict(dict(
            layer = nn.ModuleList([Layer(config) for _ in range(config.n_layers)])
        ))
        self.dropout = nn.Dropout(config.dropout)
        self.hm_head = nn.Linear(config.h_dim, config.v_size, bias=False)   # Health Modeling head
        self.embeddings.value_embeds.weight = self.hm_head.weight           # Tie weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, value_ids, phenotype_ids, attention_mask=None, labels=None, embeds=None):
        B, L = value_ids.size()
        
        # Embeddings:        
        x = self.embeddings.value_embeds(value_ids)
        x += self.embeddings.phenotype_embeds(phenotype_ids)
        x = self.dropout(x)
        if embeds is not None:
            x = torch.cat((embeds, x), dim=-2)
        
        # Transformer layers:
        for layer in self.encoder.layer:
            x = layer(x, mask=attention_mask)
        x = self.ln(x)
        
        # HM head:
        logits = self.hm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.v_size), labels.view(-1))

        return {'loss': loss, 'logits': logits, 'hidden_states': x}

    def predict(self, value_ids, phenotype_ids, bool_traits):
        with torch.no_grad():
            outputs = self.forward(value_ids, phenotype_ids)
        logits = outputs['logits']
        probs = F.softmax(logits, dim=-1)

        scores = {}
        for trait_id, trait_info in bool_traits.items():
            trait_pos = (phenotype_ids == trait_id)
            
            if not trait_pos.any():
                continue

            trait_probs = probs[trait_pos]
            positive_probs = trait_probs[:, trait_info['true_id']]
            negative_probs = trait_probs[:, trait_info['false_id']]
            score = positive_probs / (positive_probs + negative_probs)
            scores[trait_info['name']] = score
        return scores
    
    @classmethod
    def from_lightning_checkpoint(cls, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        state_dict = checkpoint['state_dict']
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        
        hparams = checkpoint.get('hyper_parameters', {})
        config = PhETConfig()
        config.update(**hparams.get('config', {}))
        model = cls(config)

        model.load_state_dict(state_dict)    
        return model
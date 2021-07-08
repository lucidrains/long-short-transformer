import torch
from torch import nn, einsum

from einops import rearrange

def exists(val):
    return val is not None

def rotate_half(x):
    x = rearrange(x, 'b n (r d) -> b n r d', r = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_emb(freqs, t):
    cos, sin = freqs
    rot_dim = cos.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * cos) + (rotate_half(t) * sin)
    return torch.cat((t, t_pass), dim = -1)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.cache = dict()
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, t, cache_key = None):
        if exists(cache_key) and cache_key in self.cache:
            return self.cache[cache_key]

        t = t.type(self.inv_freq.dtype)
        freqs = torch.einsum('i, j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)
        emb = (freqs.cos(), freqs.sin())

        if exists(cache_key):
            self.cache[cache_key] = emb

        return emb

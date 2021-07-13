from math import gcd, ceil
import functools

import torch
from torch import nn, einsum
import torch.nn.functional as F

from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def lcm(*numbers):
    return int(functools.reduce(lambda x, y: int((x * y) / gcd(x, y)), numbers, 1))

def pad_to_multiple(tensor, multiple, dim = -1, value = 0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple

    if m.is_integer():
        return tensor

    remainder = ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(tensor, (*pad_offset, 0, remainder), value=value)

def look_around(x, backward = 1, forward = 0, pad_value = -1, dim = 2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value= pad_value)
    tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
    return torch.cat(tensors, dim=dim)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class LongShortAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        causal = True,
        window_size = 128,
        pos_emb = None,
        segment_size = 16,
        r = 1,
        dropout = 0.
    ):
        super().__init__()
        assert not (causal and r >= segment_size), 'r should be less than segment size, if autoregressive'

        inner_dim = heads * dim_head
        self.scale = dim_head ** -0.5

        self.heads = heads
        self.causal = causal

        self.window_size = window_size
        self.segment_size = segment_size
        self.pad_to_multiple = window_size if not causal else lcm(window_size, segment_size)

        self.to_dynamic_proj = nn.Linear(dim_head, r, bias = False)
        self.local_norm = nn.LayerNorm(dim_head)
        self.global_norm = nn.LayerNorm(dim_head)

        self.pos_emb = default(pos_emb, RotaryEmbedding(dim_head))

        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, mask = None):
        b, n, *_, h, device, causal, w, s = *x.shape, self.heads, x.device, self.causal, self.window_size, self.segment_size

        # pad input sequence to multiples of window size (or window size and segment length if causal)

        x = pad_to_multiple(x, self.pad_to_multiple, dim = -2, value = 0.)

        # derive from variables

        padded_len = x.shape[-2]
        windows = padded_len // w
        is_padded = padded_len != n

        mask_value = -torch.finfo(x.dtype).max

        # handle mask if padding was needed and mask was not given

        if is_padded:
            mask = default(mask, torch.ones((b, n), device = device).bool())
            mask = pad_to_multiple(mask, w, dim = -1, value = False)

        # get queries, keys, values

        qkv = (self.to_q(x), self.to_kv(x))

        # get sequence range, for calculating mask

        seq_range = torch.arange(padded_len, device = device)

        # split heads

        q, kv = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), qkv)

        # rotary embedding

        if exists(self.pos_emb):
            rotary_emb = self.pos_emb(seq_range, cache_key = padded_len)
            rotary_emb = rearrange(rotary_emb, 'n d -> () n d')
            q, kv = map(lambda t: apply_rotary_emb(rotary_emb, t), (q, kv))

        # scale queries

        q = q * self.scale

        # get local queries and keys similarity scores

        window_fn = lambda t: rearrange(t, 'b (w n) d -> b w n d', n = w)
        lq, lkv = map(window_fn, (q, kv))

        lookaround_kwargs = {'backward': 1, 'forward': (0 if causal else 1)}
        lkv = look_around(lkv, **lookaround_kwargs)

        lkv = self.local_norm(lkv)
        lsim = einsum('b w i d, b w j d -> b w i j', lq, lkv)

        # prepare global key / values

        if self.causal:
            # autoregressive global attention is handled in segments
            # later on, these segments are carefully masked to prevent leakage

            gkv = rearrange(kv, 'b (n s) d -> b n s d', s = s)
            pkv = self.to_dynamic_proj(gkv)

            if exists(mask):
                pmask = rearrange(mask, 'b (n s) -> b n s', s = s)
                pkv.masked_fill_(~pmask[..., None], mask_value)

            pkv = pkv.softmax(dim = -2)

            gkv = einsum('b n s d, b n s r -> b n r d', gkv, pkv)
            gkv = rearrange(gkv, 'b n r d -> b (n r) d')
        else:
            # equation (3) in the paper

            pkv = self.to_dynamic_proj(kv)

            if exists(mask):
                pkv.masked_fill_(~mask[..., None], mask_value)

            pkv = pkv.softmax(dim = -2)

            gkv = einsum('b n d, b n r -> b r d', kv, pkv)

        # calculate global queries and keys similarity scores

        gkv = self.global_norm(gkv)
        gsim = einsum('b n d, b r d -> b n r', q, gkv)

        # concat values together (same as keys)

        gkv = repeat(gkv, 'b r d -> b w r d', w = windows)
        v = torch.cat((gkv, lkv), dim = -2)

        # masking

        buckets, i, j = lsim.shape[-3:]

        if exists(mask):
            mask = repeat(mask, 'b (w n) -> (b h) w n', n = w, h = h)
            mask = look_around(mask, pad_value = False, **lookaround_kwargs)
            mask = rearrange(mask, 'b w n -> b w () n')
            lsim.masked_fill_(~mask, mask_value)

        # mask out padding

        seq_range_windowed = rearrange(seq_range, '(w n) -> () w n', w = windows)
        pad_mask = look_around(seq_range_windowed, pad_value = -1, **lookaround_kwargs) == -1
        lsim.masked_fill_(pad_mask[:, :, None], mask_value)

        # calculate causal masking for both global and local

        if self.causal:
            g_range = rearrange(seq_range, '(n s) -> n s', s = s)
            g_range_max = g_range.amax(dim = -1)
            g_mask = seq_range[:, None] >= g_range_max[None, :]
            g_mask = rearrange(g_mask, 'i j -> () i j')
            gsim.masked_fill_(~g_mask, mask_value)

            causal_mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()
            causal_mask = repeat(causal_mask, 'i j -> () u i j', u = buckets)
            lsim.masked_fill_(causal_mask, mask_value)

        # concat local and global similarities together to ready for attention

        gsim = rearrange(gsim, 'b (w n) r -> b w n r', w = windows)
        sim = torch.cat((gsim, lsim), dim = -1)

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # aggregate values (same as keys, since tied) and project out

        out = einsum('b w i j, b w j d -> b w i d', attn, v)
        out = rearrange(out, '(b h) w n d -> b (w n) (h d)', h = h)
        out = out[:, :n]
        return self.to_out(out)

# main class

class LongShortTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        max_seq_len,
        window_size = 128,
        causal = True,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        segment_size = None,
        r = None,
        ff_dropout = 0.,
        attn_dropout = 0.
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(num_tokens, dim)
        pos_emb = RotaryEmbedding(dim_head)

        # handle autoregressive default variables differently
        # specifically, segments are only used for autoregressive case
        # r is the projected r << n in the non-autoregressive case, and the projected r per segment for the autoregressive case
        # yea, it is confusing, i know

        segment_size = default(segment_size, 16 if causal else None)
        r = default(r, 1 if causal else 128)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, LongShortAttention(dim = dim, heads = heads, dim_head = dim_head, window_size = window_size, causal = causal, pos_emb = pos_emb, segment_size = segment_size, r = r, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout))
            ]))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def forward(self, x, mask = None):
        x = self.token_emb(x)

        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x
            x = ff(x) + x

        return self.to_logits(x)

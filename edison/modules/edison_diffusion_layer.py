from collections import namedtuple

from torch import nn, einsum
from einops import rearrange


class XTAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads = 8,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, 'dim must be divisible by num_heads'
        self.scale = self.head_dim ** -0.5
        
        self.words_to_q = nn.Linear(self.dim, self.head_dim*self.num_heads, bias = False)
        self.words_to_k = nn.Linear(self.dim, self.head_dim*self.num_heads, bias = False)
        self.words_to_v = nn.Linear(self.dim, self.head_dim*self.num_heads, bias = False)
        self.position_to_q = nn.Linear(self.dim, self.head_dim*self.num_heads, bias = False)
        self.position_to_k = nn.Linear(self.dim, self.head_dim*self.num_heads, bias = False)
        self.conscious_to_q = nn.Linear(self.dim, self.head_dim*self.num_heads, bias = False)
        self.conscious_to_k = nn.Linear(self.dim, self.head_dim*self.num_heads, bias = False)
        
        self.to_out = nn.Linear(self.head_dim*self.num_heads, self.dim)
    
    # option 1
    def forward(self, x):
        words, position, conscious = x["words"], x["position"], x["conscious"]
        h = self.num_heads
        
        q_words = rearrange(self.words_to_q(words), 'b n (h d) -> b h n d', h = h)
        k_words = rearrange(self.words_to_k(words), 'b n (h d) -> b h n d', h = h)
        v_words = rearrange(self.words_to_v(words), 'b n (h d) -> b h n d', h = h)
        
        q_position = rearrange(self.position_to_q(position), 'b n (h d) -> b h n d', h = h)
        k_position = rearrange(self.position_to_k(position), 'b n (h d) -> b h n d', h = h)
        
        q_conscious = rearrange(self.conscious_to_q(conscious), 'b n (h d) -> b h n d', h = h)
        k_conscious = rearrange(self.conscious_to_k(conscious), 'b n (h d) -> b h n d', h = h)
        
        q_list = [q_words, q_position, q_conscious]
        k_list = [k_words, k_position, k_conscious]
        sim_all = 0
        for q, k in zip(q_list, k_list):
            sim = einsum('b h i j, b h j d -> b h i d', q, k) * self.scale
            sim_all += sim
        attn = sim_all.softmax(dim = -1)
        
        
        out = einsum('b h i j, b h j d -> b h i d', attn, v_words)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
    # option 2
    # TODO: c 정하기(attention mask 방식? 다른 방식?), mask 어떻게?
    def forward_2(self, x):
        words, position, conscious = x["words"], x["position"], x["conscious"]
        h = self.num_heads
        
        #c=1
        q_words_c1 = rearrange(self.words_to_q(words), 'b n (h d) -> b h n d', h = h)
        k_words_c1 = rearrange(self.words_to_k(words), 'b n (h d) -> b h n d', h = h)
        v_words_c1 = rearrange(self.words_to_v(words), 'b n (h d) -> b h n d', h = h)
        q_position_c1 = rearrange(self.position_to_q(position), 'b n (h d) -> b h n d', h = h)
        k_position_c1 = rearrange(self.position_to_k(position), 'b n (h d) -> b h n d', h = h)
        #c=0
        q_words_c0 = rearrange(self.words_to_q(words), 'b n (h d) -> b h n d', h = h)
        k_words_c0 = rearrange(self.words_to_k(words), 'b n (h d) -> b h n d', h = h)
        v_words_c0 = rearrange(self.words_to_v(words), 'b n (h d) -> b h n d', h = h)
        q_position_c0 = rearrange(self.position_to_q(position), 'b n (h d) -> b h n d', h = h)
        k_position_c0 = rearrange(self.position_to_k(position), 'b n (h d) -> b h n d', h = h)
        #c=m
        q_words_cm = rearrange(self.words_to_q(words), 'b n (h d) -> b h n d', h = h)
        k_words_cm = rearrange(self.words_to_k(words), 'b n (h d) -> b h n d', h = h)
        v_words_cm = rearrange(self.words_to_v(words), 'b n (h d) -> b h n d', h = h)
        q_position_cm = rearrange(self.position_to_q(position), 'b n (h d) -> b h n d', h = h)
        k_position_cm = rearrange(self.position_to_k(position), 'b n (h d) -> b h n d', h = h)
        
        q_word = q_words_c1 + q_words_c0 + q_words_cm
        q_position = q_position_c1 + q_position_c0 + q_position_cm
        k_word = k_words_c1 + k_words_c0 + k_words_cm
        k_position = k_position_c1 + k_position_c0 + k_position_cm
        v_words = v_words_c1 + v_words_c0 + v_words_cm
        
        q_list = [q_word, q_position]
        k_list = [k_word, k_position]
        sim_all = 0
        for q, k in zip(q_list, k_list):
            sim = einsum('b h i j, b h j d -> b h i d', q, k) * self.scale
            sim_all += sim
        attn = sim_all.softmax(dim = -1)
        
        out = einsum('b h i j, b h j d -> b h i d', attn, v_words)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.ff = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ff(x)


class Encoder(nn.Module):
    def __init__(self, dim, depth, num_heads=8):
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    nn.LayerNorm(dim),
                    XTAttention(dim, num_heads),
                    nn.LayerNorm(dim),
                    FeedForward(dim)
                ])
            )

    def forward(self, x, mask=None):
        #x must be dict, have x["words"], x["position"], x["conscious"]
        for norm1, attn, norm2, ff in self.layers:
            x = attn(norm1(x), mask=mask) + x
            x = ff(norm2(x)) + x
        return x


import torch
from torch import nn, einsum
from einops import rearrange, repeat

from layers.residual import TimeConditionedResidual, GRUGating
from layers.positional_embedding import SinusoidalPosEmb, ConsciousnessEmbedding, RelativePositionEmbedding


class XTAttention(nn.Module):
    def __init__(
        self,
        dim,
        cross_dim = None,
        num_heads = 8,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, 'dim must be divisible by num_heads'
        self.scale = self.head_dim ** -0.5
        
        self.words_to_q = nn.Linear(self.dim, self.head_dim*self.num_heads, bias = False)
        self.words_to_k = nn.Linear(self.dim if cross_dim is None else cross_dim, self.head_dim*self.num_heads, bias = False)
        self.words_to_v = nn.Linear(self.dim if cross_dim is None else cross_dim, self.head_dim*self.num_heads, bias = False)
        self.position_to_q = nn.Linear(self.dim, self.head_dim*self.num_heads, bias = False)
        self.position_to_k = nn.Linear(self.dim, self.head_dim*self.num_heads, bias = False)
        # self.position_to_k = nn.Linear(self.dim if cross_dim is None else cross_dim, self.head_dim*self.num_heads, bias = False)
        self.conscious_to_q = nn.Linear(self.dim, self.head_dim*self.num_heads, bias = False)
        self.conscious_to_k = nn.Linear(self.dim, self.head_dim*self.num_heads, bias = False)
        # self.conscious_to_k = nn.Linear(self.dim if cross_dim is None else cross_dim, self.head_dim*self.num_heads, bias = False)

        self.to_out = nn.Linear(self.head_dim*self.num_heads, self.dim)

    # option 1
    def forward(self, words, position_words, conscious_words, cross_kv=None, position_cross=None, conscious_cross=None):
        h = self.num_heads
        # print(f"words: {words.shape}, position_words: {position_words.shape}, conscious_words: {conscious_words.shape}")
        # print(f"cross_kv: {cross_kv.shape}, position_cross: {position_cross.shape}, conscious_cross: {conscious_cross.shape}")
        # print(f"words_to_q: {self.words_to_q}, words_to_k: {self.words_to_k}, words_to_v: {self.words_to_v}")
        # print(f"position_to_q: {self.position_to_q}, position_to_k: {self.position_to_k}")
        # print(f"conscious_to_q: {self.conscious_to_q}, conscious_to_k: {self.conscious_to_k}")
        q_words = rearrange(self.words_to_q(words), 'b n (h d) -> b h n d', h = h)
        k_words = rearrange(self.words_to_k(cross_kv), 'b n (h d) -> b h n d', h = h)
        v_words = rearrange(self.words_to_v(cross_kv), 'b n (h d) -> b h n d', h = h)

        q_position = rearrange(self.position_to_q(position_words), 'b n (h d) -> b h n d', h = h)
        k_position = rearrange(self.position_to_k(position_cross), 'b n (h d) -> b h n d', h = h)

        q_conscious = rearrange(self.conscious_to_q(conscious_words), 'b n (h d) -> b h n d', h = h)
        k_conscious = rearrange(self.conscious_to_k(conscious_cross), 'b n (h d) -> b h n d', h = h)
        # print(f"q_words: {q_words.shape}, k_words: {k_words.shape}, v_words: {v_words.shape}")
        # print(f"q_position: {q_position.shape}, k_position: {k_position.shape}")
        # print(f"q_conscious: {q_conscious.shape}, k_conscious: {k_conscious.shape}")
        q_list = [q_words, q_position, q_conscious]
        # if cross_kv == None:
        k_list = [k_words, k_position, k_conscious]
        for q, k in zip(q_list, k_list):
            sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
            try:
                sim_all = sim_all + sim
            except:
                sim_all = sim
        attn = sim_all.softmax(dim=-1)
        # else:
        #     for q in q_list:
        #         sim = einsum('b h i d, b h j d -> b h i j', q, k_words) * self.scale
        #         try:
        #             sim_all = sim_all + sim
        #         except:
        #             sim_all = sim
        #     attn = sim_all.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v_words)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    # option 2
    # TODO: c 정하기(attention mask 방식? 다른 방식?), mask 어떻게?
    def forward_2(self, x):
        words, position, conscious = x['words'], x['position'], x['conscious']
        h = self.num_heads

        # c=1
        q_words_c1 = rearrange(self.words_to_q(words), 'b n (h d) -> b h n d', h = h)
        k_words_c1 = rearrange(self.words_to_k(words), 'b n (h d) -> b h n d', h = h)
        v_words_c1 = rearrange(self.words_to_v(words), 'b n (h d) -> b h n d', h = h)
        q_position_c1 = rearrange(self.position_to_q(position), 'b n (h d) -> b h n d', h = h)
        k_position_c1 = rearrange(self.position_to_k(position), 'b n (h d) -> b h n d', h = h)
        # c=0
        q_words_c0 = rearrange(self.words_to_q(words), 'b n (h d) -> b h n d', h = h)
        k_words_c0 = rearrange(self.words_to_k(words), 'b n (h d) -> b h n d', h = h)
        v_words_c0 = rearrange(self.words_to_v(words), 'b n (h d) -> b h n d', h = h)
        q_position_c0 = rearrange(self.position_to_q(position), 'b n (h d) -> b h n d', h = h)
        k_position_c0 = rearrange(self.position_to_k(position), 'b n (h d) -> b h n d', h = h)
        # c=m
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


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        cross_dim = None,
        num_heads = 8
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, 'dim must be divisible by num_heads'
        self.scale = self.head_dim ** -0.5
        
        self.words_to_q = nn.Linear(self.dim, self.head_dim*self.num_heads, bias = False)
        self.words_to_k = nn.Linear(self.dim if cross_dim == None else cross_dim, self.head_dim*self.num_heads, bias = False)
        self.words_to_v = nn.Linear(self.dim if cross_dim == None else cross_dim, self.head_dim*self.num_heads, bias = False)
        
        self.to_out = nn.Linear(self.head_dim*self.num_heads, self.dim)

    def forward(self, words, cross_kv=None):
        words_kv = words if cross_kv is None else cross_kv
        h = self.num_heads
        q = rearrange(self.words_to_q(words), 'b n (h d) -> b h n d', h = h)
        k = rearrange(self.words_to_k(words_kv), 'b n (h d) -> b h n d', h = h)
        v = rearrange(self.words_to_v(words_kv), 'b n (h d) -> b h n d', h = h)
        
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, ff_mult=4, dropout=0.):
        super().__init__()
        inner_dim = int(dim * ff_mult)
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
    def __init__(self, dim, cross_dim, depth, num_heads=8, ff_mult=4, max_seq_len=64, cross_max_seq_len=32, is_context_diffusion=False, num_dense_connections=3):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.ff_mult = ff_mult
        self.is_context_diffusion = is_context_diffusion
        self.num_dense_connections = num_dense_connections
        self.layers = nn.ModuleList()
        for _ in range(depth-1):
            self.layers.append(
                nn.ModuleList([
                    nn.LayerNorm(dim),
                    XTAttention(dim, cross_dim, num_heads=num_heads),
                    GRUGating(dim),
                    nn.LayerNorm(dim),
                    XTAttention(dim, num_heads=num_heads),
                    GRUGating(dim),
                    nn.LayerNorm(dim),
                    FeedForward(dim, ff_mult),
                    TimeConditionedResidual(dim*ff_mult, dim),
                ])
            )
        self.last_layers = nn.ModuleList([
            nn.LayerNorm(dim),
            XTAttention(dim, num_heads=num_heads),
            GRUGating(dim),
            nn.LayerNorm(dim),
            XTAttention(dim, num_heads=num_heads),
            GRUGating(dim),
            nn.LayerNorm(dim),
            FeedForward(dim, ff_mult),
            TimeConditionedResidual(dim*ff_mult, dim),
        ])
        self.project_embedding_to_position = self._get_continuous_position_layer()
        self.project_embedding_to_conscious = self._get_conscious_layer()
        self.relative_position_layer = self._get_relative_position_layer(max_seq_len)
        self.cross_relative_position_layer = self._get_relative_position_layer(cross_max_seq_len)
        self.proj_dense_connection = nn.Linear(dim*2, dim)

    def _get_continuous_position_layer(self):
        return torch.nn.Sequential(
            SinusoidalPosEmb(self.dim),
            nn.Linear(self.dim, self.dim * self.ff_mult),
            nn.GELU(),
            nn.Linear(self.dim * self.ff_mult, self.dim),
        )

    def _get_relative_position_layer(self, max_seq_len:int):
        return RelativePositionEmbedding(max_seq_len=max_seq_len, hidden_dim=self.dim)

    def _get_conscious_layer(self):
        return ConsciousnessEmbedding(
            dim=self.dim,
            num_flag=2,
        )

    def _get_xt_data(self, words, attention_mask_words, attention_mask_cross):
        # 여기서 position은 CPE(continuous position embedding)
        position_input = torch.arange(words.shape[1], device=words.device)
        position_input = repeat(position_input, 'n -> b n', b=words.shape[0])
        # make buffer word position to zero
        if self.is_context_diffusion:
            position_input = position_input+1
        else:
            position_input = (position_input+1) * attention_mask_words
        position = self.project_embedding_to_position(position_input)
        conscious_words = self.project_embedding_to_conscious(attention_mask=attention_mask_words)
        conscious_cross = self.project_embedding_to_conscious(attention_mask=attention_mask_cross)
        return position, conscious_words, conscious_cross

    def forward(self, words, cross_kv, attention_mask_words, attention_mask_cross, time_emb=None):
        # 여기서 position = RPE (고정 - 마지막 2개 레이어도 마찬가지로 사용)
        # conscious도 고정
        rpe_words = self.relative_position_layer(words.shape[0], words.shape[1], words.device)
        rpe_cross = self.cross_relative_position_layer(cross_kv.shape[0], cross_kv.shape[1], cross_kv.device)
        cpe, conscious_words, conscious_cross = self._get_xt_data(words, attention_mask_words, attention_mask_cross)
        # 마지막 2개 레이어: CPE 추가, hidden state에 더해서 사용
        hidden_states = []
        for i, layers in enumerate(self.layers):
            words = self._forward_layers(layers, words, cross_kv, rpe_words, rpe_cross, conscious_words, conscious_cross, time_emb)
            # Dense connection
            self._maybe_dense_connection(i, words, hidden_states)
        words_plus_rpe = words + cpe
        words = self._forward_layers(self.last_layers, words_plus_rpe, words, rpe_words, rpe_words, conscious_words, conscious_words, time_emb)
        words = self._forward_layers(self.last_layers, words, words, rpe_words, rpe_words, conscious_words, conscious_words, time_emb)
        return words

    def _forward_layers(self, layers, words, cross_kv, rpe_words, rpe_cross, conscious_words, conscious_cross, time_emb=None):
        (norm1, cross_attn, cross_attn_residual,
        norm2, self_attn, self_attn_residual,
        norm3, ff, ff_residual) = layers
        residual = words
        # print('words 1:', words.shape)
        words = cross_attn(norm1(words), rpe_words, conscious_words, cross_kv, rpe_cross, conscious_cross)
        # print('words 2:', words.shape)
        words = cross_attn_residual(words, residual)
        # print('words 3:', words.shape)
        residual = words
        words = self_attn(norm2(words), rpe_words, conscious_words, norm2(words), rpe_words, conscious_words)
        words = self_attn_residual(words, residual)
        residual = words
        words = ff(norm3(words))
        words = ff_residual(words, residual, time_emb)
        return words
    
    def _maybe_dense_connection(self, idx, words, hidden_states:list):
        if idx < self.num_dense_connections:
            hidden_states.append(words)
            return words, hidden_states
        elif idx >= (len(self.layers) - self.num_dense_connections):
            words = self.proj_dense_connection(torch.cat([words, hidden_states.pop(-1)], dim=-1))
            return words, hidden_states
        else:
            return words, hidden_states

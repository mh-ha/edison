import torch
from torch import nn, Tensor, einsum
from einops import rearrange, repeat

from edison.layers.base import BaseEncoder
from edison.layers.residual import TimeConditionedResidual, GRUGating
from edison.layers.positional_embedding import SinusoidalPosEmb, ConsciousnessEmbedding, RelativePositionEmbedding


class XTAttention(nn.Module):
    def __init__(
        self,
        dim,
        cross_dim=None,
        num_heads=12,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, 'dim must be divisible by num_heads'
        self.scale = self.head_dim ** -0.5

        self.words_to_q = nn.Linear(self.dim, self.head_dim*self.num_heads, bias=False)
        self.words_to_k = nn.Linear(self.dim if cross_dim is None else cross_dim, self.head_dim*self.num_heads, bias=False)
        self.words_to_v = nn.Linear(self.dim if cross_dim is None else cross_dim, self.head_dim*self.num_heads, bias=False)
        self.position_to_q = nn.Linear(self.dim, self.head_dim*self.num_heads, bias=False)
        self.position_to_k = nn.Linear(self.dim, self.head_dim*self.num_heads, bias=False)
        self.conscious_to_q = nn.Linear(self.dim, self.head_dim*self.num_heads, bias=False)
        self.conscious_to_k = nn.Linear(self.dim, self.head_dim*self.num_heads, bias=False)
        if cross_dim is not None:
            # use latents instead of conscious
            self.latent_conscious_k = nn.Parameter(torch.randn(self.head_dim*self.num_heads))
        else:
            self.latent_conscious_k = None

        self.to_out = nn.Linear(self.head_dim*self.num_heads, self.dim)

    # option 1
    def forward(self, words, position_words, conscious_words, cross_kv=None, position_cross=None, conscious_cross=None):
        # print(f"[XTAttention.forward.entry] words: {words.shape}, context: {cross_kv.shape}")
        h = self.num_heads
        q_words = rearrange(self.words_to_q(words), 'b n (h d) -> b h n d', h=h)
        k_words = rearrange(self.words_to_k(cross_kv), 'b n (h d) -> b h n d', h=h)
        v_words = rearrange(self.words_to_v(cross_kv), 'b n (h d) -> b h n d', h=h)
        q_position = rearrange(self.position_to_q(position_words), 'b n (h d) -> b h n d', h=h)
        k_position = rearrange(self.position_to_k(position_cross), 'b n (h d) -> b h n d', h=h)
        q_conscious = rearrange(self.conscious_to_q(conscious_words), 'b n (h d) -> b h n d', h=h)
        if self.latent_conscious_k is not None:
            k_conscious = repeat(self.latent_conscious_k, 'd -> b n d', b=cross_kv.shape[0], n=cross_kv.shape[1])
            k_conscious = rearrange(self.conscious_to_k(k_conscious), 'b n (h d) -> b h n d', h=h)
        else:
            k_conscious = rearrange(self.conscious_to_k(conscious_cross), 'b n (h d) -> b h n d', h=h)
        # print(f"[XTAttention.forward.qkv] q_words: {q_words.shape}, k_words: {k_words.shape}, v_words: {v_words.shape}")
        # print(f"[XTAttention.forward.qk_position] q_position: {q_position.shape}, k_position: {k_position.shape}")
        # print(f"[XTAttention.forward.qk_conscious] q_conscious: {q_conscious.shape}, k_conscious: {k_conscious.shape}")

        q_list = [q_words, q_position, q_conscious]
        k_list = [k_words, k_position, k_conscious]
        score = None
        for q in q_list:
            for k in k_list:
                sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
                # print(f"[XTAttention.forward.sim] sim: {sim.shape}")
                score = score + sim if score is not None else sim
        attn = score.softmax(dim=-1)
        # print(f"[XTAttention.forward.attention] attn: {attn.shape}")

        out = einsum('b h i j, b h j d -> b h i d', attn, v_words)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        # print(f"[XTAttention.forward.exit] out: {out.shape}")
        return out


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        cross_dim=None,
        num_heads=12
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, 'dim must be divisible by num_heads'
        self.scale = self.head_dim ** -0.5

        self.words_to_q = nn.Linear(self.dim, self.head_dim*self.num_heads, bias=False)
        self.words_to_k = nn.Linear(self.dim if cross_dim is None else cross_dim, self.head_dim*self.num_heads, bias=False)
        self.words_to_v = nn.Linear(self.dim if cross_dim is None else cross_dim, self.head_dim*self.num_heads, bias=False)

        self.to_out = nn.Linear(self.head_dim*self.num_heads, self.dim)

    def forward(self, words, rpe_words=None, cross_kv=None, rpe_cross=None):
        # print(f"[Attention.forward.entry] context: {words.shape}, words: {cross_kv.shape}")
        words_kv = words if cross_kv is None else cross_kv
        if rpe_words is not None:
            rpe_cross = rpe_words if rpe_cross is None else rpe_cross
            words = words + rpe_words
            words_kv = words_kv + rpe_cross
        h = self.num_heads
        q = rearrange(self.words_to_q(words), 'b n (h d) -> b h n d', h=h)
        k = rearrange(self.words_to_k(words_kv), 'b n (h d) -> b h n d', h=h)
        v = rearrange(self.words_to_v(words_kv), 'b n (h d) -> b h n d', h=h)
        score = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = score.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # print(f"[Attention.forward.exit] out: {out.shape}")
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


class Encoder(BaseEncoder):
    """
    embedding은 그대로
    context는 LD4LG와 동일한 구조로
    둘의 중간 결과값을 cross attention때 합침
    마지막 output은 embedding output만
    """
    def __init__(
        self,
        internal_dim: int,
        depth: int,
        num_heads: int = 8,
        ff_mult: int = 4,
        max_seq_len: int = 64,
        context_max_seq_len: int = 32,
        num_dense_connections: int = 3
    ):
        super().__init__(
            internal_dim=internal_dim,
            depth=depth,
        )
        self.ff_mult = ff_mult
        self.num_dense_connections = num_dense_connections

        # Embedding
        self.layers_for_embedding = nn.ModuleList()
        for _ in range(depth-1):
            self.layers_for_embedding.append(
                nn.ModuleList([
                    nn.LayerNorm(internal_dim),
                    XTAttention(internal_dim, internal_dim, num_heads=num_heads),
                    GRUGating(internal_dim),
                    nn.LayerNorm(internal_dim),
                    XTAttention(internal_dim, num_heads=num_heads),
                    GRUGating(internal_dim),
                    nn.LayerNorm(internal_dim),
                    FeedForward(internal_dim, ff_mult),
                    TimeConditionedResidual(internal_dim*ff_mult, internal_dim),
                ])
            )
        self.last_layers_for_embedding = nn.ModuleList([
            nn.LayerNorm(internal_dim),
            XTAttention(internal_dim, num_heads=num_heads),
            GRUGating(internal_dim),
            nn.LayerNorm(internal_dim),
            XTAttention(internal_dim, num_heads=num_heads),
            GRUGating(internal_dim),
            nn.LayerNorm(internal_dim),
            FeedForward(internal_dim, ff_mult),
            TimeConditionedResidual(internal_dim*ff_mult, internal_dim),
        ])

        # Context
        self.layers_for_context = nn.ModuleList()
        for _ in range(depth-1):
            self.layers_for_context.append(
                nn.ModuleList([
                    nn.LayerNorm(internal_dim),
                    Attention(internal_dim, internal_dim, num_heads=num_heads),
                    GRUGating(internal_dim),
                    nn.LayerNorm(internal_dim),
                    Attention(internal_dim, num_heads=num_heads),
                    GRUGating(internal_dim),
                    nn.LayerNorm(internal_dim),
                    FeedForward(internal_dim, ff_mult),
                    TimeConditionedResidual(internal_dim*ff_mult, internal_dim),
                ])
            )

        self.project_embedding_to_position = self._get_continuous_position_layer()
        self.project_embedding_to_conscious = self._get_conscious_layer()
        self.embedding_relative_position_layer = self._get_relative_position_layer(max_seq_len)
        self.context_relative_position_layer = self._get_relative_position_layer(context_max_seq_len)
        self.proj_dense_connection = nn.Linear(internal_dim*2, internal_dim)

    def _get_continuous_position_layer(self):
        return torch.nn.Sequential(
            SinusoidalPosEmb(self.internal_dim),
            nn.Linear(self.internal_dim, self.internal_dim * self.ff_mult),
            nn.GELU(),
            nn.Linear(self.internal_dim * self.ff_mult, self.internal_dim),
        )

    def _get_relative_position_layer(self, max_seq_len: int):
        return RelativePositionEmbedding(max_seq_len=max_seq_len, hidden_dim=self.internal_dim)

    def _get_conscious_layer(self):
        return ConsciousnessEmbedding(
            dim=self.internal_dim,
            num_flag=2,
        )

    def _get_xt_data(self, latent, attention_mask):
        # 여기서 position은 CPE(continuous position embedding)
        position_input = torch.arange(latent.shape[1], device=latent.device)
        position_input = repeat(position_input, 'n -> b n', b=latent.shape[0])
        # make buffer word position to zero
        position_input = (position_input+1) * attention_mask
        position = self.project_embedding_to_position(position_input)
        conscious_words = self.project_embedding_to_conscious(attention_mask=attention_mask)
        return position, conscious_words

    def forward(
        self,
        latent: Tensor,
        context: Tensor,
        attention_mask: Tensor,
        time_emb: Tensor,
    ) -> Tensor:
        # print(f"[Encoder.forward.entry] words: {words.shape}, cross_kv: {cross_kv.shape}")
        # 여기서 position = RPE (고정 - 마지막 2개 레이어도 마찬가지로 사용)
        # conscious도 고정
        rpe_words = self.embedding_relative_position_layer(latent.shape[0], latent.shape[1], latent.device)
        rpe_cross = self.context_relative_position_layer(context.shape[0], context.shape[1], context.device)
        cpe, conscious_words = self._get_xt_data(latent, attention_mask)
        # 마지막 2개 레이어: CPE 추가, hidden state에 더해서 사용
        emb_hidden_states = []
        context_hidden_states = []
        for i, (emb_layers, context_layers) in enumerate(zip(self.layers_for_embedding, self.layers_for_context)):
            latent, context = self._forward_layers(
                emb_layers, context_layers, latent, context, rpe_words, rpe_cross, conscious_words, time_emb
            )
            # Dense connection
            latent, emb_hidden_states = self._maybe_dense_connection(i, latent, emb_hidden_states)
            context, context_hidden_states = self._maybe_dense_connection(i, context, context_hidden_states)
        words_plus_rpe = latent + cpe
        latent = self._forward_last_layers(
            self.last_layers_for_embedding, words_plus_rpe, latent, rpe_words, conscious_words, time_emb
        )
        latent = self._forward_last_layers(
            self.last_layers_for_embedding, latent, latent, rpe_words, conscious_words, time_emb
        )
        return latent

    def _forward_layers(self, emb_layers, context_layers, latent, context, rpe_words, rpe_cross, conscious_words, time_emb):
        # TODO: emb_layers, context_layers 통합하여 동시에 계산하도록 함으로써 속도 개선
        (
            emb_norm1, emb_cross_attn, emb_cross_attn_residual,
            emb_norm2, emb_self_attn, emb_self_attn_residual,
            emb_norm3, emb_ff, emb_ff_residual
        ) = emb_layers
        (
            context_norm1, context_cross_attn, context_cross_attn_residual,
            context_norm2, context_self_attn, context_self_attn_residual,
            context_norm3, context_ff, context_ff_residual
        ) = context_layers

        emb_residual = latent
        context_residual = context
        latent = emb_cross_attn(emb_norm1(latent), rpe_words, conscious_words, context, rpe_cross)
        context = context_cross_attn(context_norm1(context), rpe_cross, latent, rpe_words)
        latent = emb_cross_attn_residual(latent, emb_residual)
        context = context_cross_attn_residual(context, context_residual)

        emb_residual = latent
        context_residual = context
        latent = emb_self_attn(emb_norm2(latent), rpe_words, conscious_words, emb_norm2(latent), rpe_words, conscious_words)
        context = context_self_attn(context_norm2(context), rpe_cross, context_norm2(context), rpe_cross)
        latent = emb_self_attn_residual(latent, emb_residual)
        context = context_self_attn_residual(context, context_residual)

        emb_residual = latent
        context_residual = context
        latent = emb_ff(emb_norm3(latent))
        context = context_ff(context_norm3(context))
        latent = emb_ff_residual(latent, emb_residual, time_emb)
        context = context_ff_residual(context, context_residual, time_emb)
        return latent, context

    def _forward_last_layers(self, emb_layers, latent, context, rpe_words, conscious_words, time_emb):
        (
            emb_norm1, emb_cross_attn, emb_cross_attn_residual,
            emb_norm2, emb_self_attn, emb_self_attn_residual,
            emb_norm3, emb_ff, emb_ff_residual
        ) = emb_layers
        emb_residual = latent
        latent = emb_cross_attn(emb_norm1(latent), rpe_words, conscious_words, emb_norm1(context), rpe_words, conscious_words)
        latent = emb_cross_attn_residual(latent, emb_residual)
        emb_residual = latent
        latent = emb_self_attn(emb_norm2(latent), rpe_words, conscious_words, emb_norm2(latent), rpe_words, conscious_words)
        latent = emb_self_attn_residual(latent, emb_residual)
        emb_residual = latent
        latent = emb_ff(emb_norm3(latent))
        latent = emb_ff_residual(latent, emb_residual, time_emb)
        return latent

    def _maybe_dense_connection(self, idx, words, hidden_states: list):
        if idx < self.num_dense_connections:
            hidden_states.append(words)
            return words, hidden_states
        elif idx >= (len(self.layers_for_embedding) - self.num_dense_connections):
            words = self.proj_dense_connection(torch.cat([words, hidden_states.pop(-1)], dim=-1))
            return words, hidden_states
        else:
            return words, hidden_states


class BaselineEncoder(BaseEncoder):
    """
    LD4LG와 동일한 구조
    """
    def __init__(
        self,
        internal_dim: int,
        depth: int,
        num_heads: int = 8,
        ff_mult: int = 4,
        max_seq_len: int = 64,
        num_dense_connections: int = 3
    ):
        super().__init__(
            internal_dim=internal_dim,
            depth=depth,
        )
        self.ff_mult = ff_mult
        self.num_dense_connections = num_dense_connections

        # Embedding
        self.layers = nn.ModuleList()
        for _ in range(depth-1):
            self.layers.append(
                nn.ModuleList([
                    nn.LayerNorm(internal_dim),
                    Attention(internal_dim, num_heads=num_heads),
                    GRUGating(internal_dim),
                    nn.LayerNorm(internal_dim),
                    FeedForward(internal_dim, ff_mult),
                    TimeConditionedResidual(internal_dim*ff_mult, internal_dim),
                ])
            )

        self.proj_dense_connection = nn.Linear(internal_dim*2, internal_dim)

    def forward(
        self,
        latent: Tensor,
        attention_mask: Tensor,
        time_emb: Tensor,
        **kwargs,
    ) -> Tensor:
        emb_hidden_states = []

        for i, layers in enumerate(self.layers):
            latent = self._forward_layers(
                layers, latent, time_emb, rpe=None
            )
            # Dense connection
            latent, emb_hidden_states = self._maybe_dense_connection(i, latent, emb_hidden_states)
        return latent

    def _forward_layers(self, layers, latent, time_emb, rpe=None):
        (
            emb_norm2, emb_self_attn_2, emb_self_attn_residual_2,
            emb_norm3, emb_ff, emb_ff_residual
        ) = layers

        emb_residual = latent
        latent = emb_self_attn_2(emb_norm2(latent), rpe)
        latent = emb_self_attn_residual_2(latent, emb_residual)

        emb_residual = latent
        latent = emb_ff(emb_norm3(latent))
        latent = emb_ff_residual(latent, emb_residual, time_emb)
        return latent

    def _maybe_dense_connection(self, idx, words, hidden_states: list):
        if idx < self.num_dense_connections:
            hidden_states.append(words)
            return words, hidden_states
        elif idx >= (len(self.layers) - self.num_dense_connections):
            words = self.proj_dense_connection(torch.cat([words, hidden_states.pop(-1)], dim=-1))
            return words, hidden_states
        else:
            return words, hidden_states

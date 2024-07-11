import math
from functools import partial, wraps
from inspect import isfunction
from collections import namedtuple

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, einsum


class XTAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_layers = 12,
        head_dim = 64,
        latent_dim = 64,
        ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.latent_dim = latent_dim
        
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5
        
        self.words_to_q = nn.Linear(
            self.dim,
            self.head_dim*self.num_heads,
            bias = False,
            )
        self.words_to_k = nn.Linear(
            self.dim,
            self.head_dim*self.num_heads,
            bias = False,
            )
        self.words_to_v = nn.Linear(
            self.dim,
            self.head_dim*self.num_heads,
            bias = False,
            )
        self.position_to_q = nn.Linear(
            self.dim,
            self.head_dim*self.num_heads,
            bias = False,
            )
        self.position_to_k = nn.Linear(
            self.dim,
            self.head_dim*self.num_heads,
            bias = False,
            )
        self.conscious_to_q = nn.Linear(
            self.dim,
            self.head_dim*self.num_heads,
            bias = False,
            )
        self.conscious_to_k = nn.Linear(
            self.dim,
            self.head_dim*self.num_heads,
            bias = False,
            )
        
        self.to_out = nn.Linear(
            self.head_dim*self.num_heads,
            self.dim,
            )
    
    def forward(self, words, position, conscious):
        h = self.num_heads
        
        q_words = rearrange(self.words_to_q(words), 'b n (h d) -> b h n d', h = h)
        k_words = rearrange(self.words_to_k(words), 'b n (h d) -> b h n d', h = h)
        v_words = rearrange(self.words_to_v(words), 'b n (h d) -> b h n d', h = h)
        
        q_position = rearrange(self.position_to_q(position), 'b n (h d) -> b h n d', h = h)
        k_position = rearrange(self.position_to_k(position), 'b n (h d) -> b h n d', h = h)
        
        q_conscious = rearrange(self.conscious_to_q(conscious), 'b n (h d) -> b h n d', h = h)
        k_conscious = rearrange(self.conscious_to_k(conscious), 'b n (h d) -> b h n d', h = h)
        
        q = q_words + q_position + q_conscious
        k = k_words + k_position + k_conscious
        v = v_words
        
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)
        
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
    
    
class XTAttentionLayers(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads = 8,
        causal = False,
        cross_attend = False,
        only_cross = False,
        use_scalenorm = False,
        use_rmsnorm = False,
        alibi_pos_bias = False,
        alibi_num_heads = None,
        alibi_learned = False,
        rel_pos_bias = False,
        rel_pos_num_buckets = 32,
        rel_pos_max_distance = 128,
        dynamic_pos_bias = False,
        dynamic_pos_bias_log_distance = False,
        dynamic_pos_bias_mlp_depth = 2,
        dynamic_pos_bias_norm = False,
        position_infused_attn = False,
        rotary_pos_emb = False,
        rotary_emb_dim = None,
        custom_layers = None,
        sandwich_coef = None,
        par_ratio = None,
        residual_attn = False,
        cross_residual_attn = False,
        macaron = False,
        pre_norm = True,
        gate_residual = False,
        scale_residual = False,
        scale_residual_constant = 1.,
        deepnorm = False,
        shift_tokens = 0,
        sandwich_norm = False,
        zero_init_branch_output = False,
        time_emb_dim = None,
        num_dense_connections = 0,
        **kwargs
    ):
        super().__init__()
        ff_kwargs, kwargs = groupby_prefix_and_trim('ff_', kwargs)
        attn_kwargs, kwargs = groupby_prefix_and_trim('attn_', kwargs)

        dim_head = attn_kwargs.get('dim_head', DEFAULT_DIM_HEAD)

        self.dim = dim
        self.depth = depth
        self.layers = nn.ModuleList([])
        self.num_dense_connections = num_dense_connections

        self.has_pos_emb = position_infused_attn or rel_pos_bias or rotary_pos_emb
        self.pia_pos_emb = FixedPositionalEmbedding(dim) if position_infused_attn else None

        rotary_emb_dim = max(default(rotary_emb_dim, dim_head // 2), 32)
        self.rotary_pos_emb = RotaryEmbedding(rotary_emb_dim) if rotary_pos_emb else None

        assert not (alibi_pos_bias and rel_pos_bias), 'you can only choose Alibi positional bias or T5 relative positional bias, not both'
        assert rel_pos_num_buckets <= rel_pos_max_distance, 'number of relative position buckets must be less than the relative position max distance'

        # relative positional bias

        self.rel_pos = None
        if rel_pos_bias:
            self.rel_pos = RelativePositionBias(scale = dim_head ** 0.5, causal = causal, heads = heads, num_buckets = rel_pos_num_buckets, max_distance = rel_pos_max_distance)
        elif dynamic_pos_bias:
            self.rel_pos = DynamicPositionBias(dim = dim // 4, heads = heads, log_distance = dynamic_pos_bias_log_distance, depth = dynamic_pos_bias_mlp_depth, norm = dynamic_pos_bias_norm)
        elif alibi_pos_bias:
            alibi_num_heads = default(alibi_num_heads, heads)
            assert alibi_num_heads <= heads, 'number of ALiBi heads must be less than the total number of heads'
            alibi_pos_klass = LearnedAlibiPositionalBias if alibi_learned else AlibiPositionalBias
            self.rel_pos = alibi_pos_klass(heads = alibi_num_heads)

        # determine deepnorm and residual scale

        if deepnorm:
            assert scale_residual_constant == 1, 'scale residual constant is being overridden by deep norm settings'
            pre_norm = sandwich_norm = False
            scale_residual = True
            scale_residual_constant = (2 * depth) ** 0.25

        assert not (not pre_norm and sandwich_norm), 'sandwich norm cannot be used when not using prenorm'
        self.pre_norm = pre_norm
        self.sandwich_norm = sandwich_norm

        self.residual_attn = residual_attn
        self.cross_residual_attn = cross_residual_attn
        self.cross_attend = cross_attend

        norm_class = ScaleNorm if use_scalenorm else nn.LayerNorm
        norm_class = RMSNorm if use_rmsnorm else norm_class
        norm_fn = partial(norm_class, dim)

        if cross_attend and not only_cross:
            default_block = ('a', 'c', 'f')
        elif cross_attend and only_cross:
            default_block = ('c', 'f')
        else:
            default_block = ('a', 'f')

        if macaron:
            default_block = ('f',) + default_block

        # zero init

        if zero_init_branch_output:
            attn_kwargs = {**attn_kwargs, 'zero_init_output':  True}
            ff_kwargs = {**ff_kwargs, 'zero_init_output':  True}

        # calculate layer block order

        if exists(custom_layers):
            layer_types = custom_layers
        elif exists(par_ratio):
            par_depth = depth * len(default_block)
            assert 1 < par_ratio <= par_depth, 'par ratio out of range'
            default_block = tuple(filter(not_equals('f'), default_block))
            par_attn  = par_depth // par_ratio
            depth_cut = par_depth * 2 // 3  # 2 / 3 attention layer cutoff suggested by PAR paper
            par_width = (depth_cut + depth_cut // par_attn) // par_attn
            assert len(default_block) <= par_width, 'default block is too large for par_ratio'
            par_block = default_block + ('f',) * (par_width - len(default_block))
            par_head = par_block * par_attn
            layer_types = par_head + ('f',) * (par_depth - len(par_head))
        elif exists(sandwich_coef):
            assert sandwich_coef > 0 and sandwich_coef <= depth, 'sandwich coefficient should be less than the depth'
            layer_types = ('a',) * sandwich_coef + default_block * (depth - sandwich_coef) + ('f',) * sandwich_coef
        else:
            layer_types = default_block * depth

        self.layer_types = layer_types
        self.num_attn_layers = len(list(filter(equals('a'), layer_types)))

        # calculate token shifting

        shift_tokens = cast_tuple(shift_tokens, len(layer_types))

        self.scale_shift = exists(time_emb_dim)


        # iterate and construct layers

        for ind, (layer_type, layer_shift_tokens) in enumerate(zip(self.layer_types, shift_tokens)):
            is_last_layer = ind == (len(self.layer_types) - 1)

            if layer_type == 'a':
                layer = Attention(dim, heads = heads, causal = causal, **attn_kwargs)
            elif layer_type == 'c':
                layer = Attention(dim, heads = heads, **attn_kwargs)
            elif layer_type == 'f':
                layer = FeedForward(dim, **ff_kwargs)
                layer = layer if not macaron else Scale(0.5, layer)
            else:
                raise Exception(f'invalid layer type {layer_type}')

            if layer_shift_tokens > 0:
                shift_range_upper = layer_shift_tokens + 1
                shift_range_lower = -layer_shift_tokens if not causal else 0
                layer = ShiftTokens(range(shift_range_lower, shift_range_upper), layer)

            if self.scale_shift and layer_type in ['f']:
                residual = TimeConditionedResidual(time_emb_dim, dim)
            else:
                residual_fn = GRUGating if gate_residual else Residual
                residual = residual_fn(dim, scale_residual = scale_residual, scale_residual_constant = scale_residual_constant)

            pre_branch_norm = norm_fn() if pre_norm else None
            post_branch_norm = norm_fn() if sandwich_norm or (self.scale_shift and layer_type in ['f']) else None
            post_main_norm = norm_fn() if not pre_norm and not is_last_layer else None

            norms = nn.ModuleList([
                pre_branch_norm,
                post_branch_norm,
                post_main_norm
            ])

            self.layers.append(nn.ModuleList([
                norms,
                layer,
                residual
            ]))
        
        self.dense_projections = nn.ModuleList([nn.Linear(dim*2, dim) for _ in range(num_dense_connections)])

        if deepnorm:
            init_gain = (8 * depth) ** -0.25
            deepnorm_init(self, init_gain)

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None,
        attn_mask = None,
        mems = None,
        time_emb = None,
        return_hiddens = False
    ):
        assert not (self.cross_attend ^ exists(context)), 'context must be passed in if cross_attend is set to True'

        hiddens = []
        intermediates = []
        prev_attn = None
        prev_cross_attn = None

        mems = mems.copy() if exists(mems) else [None] * self.num_attn_layers

        rotary_pos_emb = None
        if exists(self.rotary_pos_emb):
            max_rotary_emb_length = max(list(map(lambda m: (m.shape[1] if exists(m) else 0) + x.shape[1], mems)))
            rotary_pos_emb = self.rotary_pos_emb(max_rotary_emb_length, x.device)

        dense_hiddens = []
        attn_idx = 0
        for ind, (layer_type, (norm, block, residual_fn)) in enumerate(zip(self.layer_types, self.layers)):
            is_last = ind == (len(self.layers) - 1)

            if layer_type == 'a':
                dense_idx = attn_idx - (self.num_attn_layers - self.num_dense_connections)
                if dense_idx >= 0:
                    assert len(dense_hiddens) > 0, 'dense connections must be in order'
                    x = self.dense_projections[dense_idx](torch.cat([x, dense_hiddens.pop()], dim=-1))
                attn_idx += 1
                if return_hiddens:
                    hiddens.append(x)
                layer_mem = mems.pop(0) if mems else None

            residual = x

            pre_branch_norm, post_branch_norm, post_main_norm = norm

            if exists(pre_branch_norm):
                x = pre_branch_norm(x)

            if layer_type == 'a':
                out, inter = block(x, mask = mask, attn_mask = attn_mask, sinusoidal_emb = self.pia_pos_emb, rel_pos = self.rel_pos, rotary_pos_emb = rotary_pos_emb, prev_attn = prev_attn, mem = layer_mem)
            elif layer_type == 'c':
                out, inter = block(x, context = context, mask = mask, context_mask = context_mask, prev_attn = prev_cross_attn)
            elif layer_type == 'f':
                out = block(x)

            if exists(post_branch_norm):
                out = post_branch_norm(out)

            if self.scale_shift and layer_type in ['f']:
                x = residual_fn(out, residual, time_emb)
            else:
                x = residual_fn(out, residual)


            if layer_type in ('a', 'c') and return_hiddens:
                intermediates.append(inter)

            if layer_type == 'a' and self.residual_attn:
                prev_attn = inter.pre_softmax_attn
            elif layer_type == 'c' and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn

            if exists(post_main_norm):
                x = post_main_norm(x)
            
            if layer_type == 'f' and len(dense_hiddens) < self.num_dense_connections:
                dense_hiddens.append(x)

        if return_hiddens:
            intermediates = LayerIntermediates(
                hiddens = hiddens,
                attn_intermediates = intermediates
            )

            return x, intermediates

        return x
from __future__ import annotations

import copy
import math
from collections.abc import Callable
from typing import Optional, Union

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, EncoderDecoderCache
from transformers.masking_utils import create_bidirectional_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutput,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack

from transformers.models.marian.configuration_marian import MarianConfig
from transformers.models.marian.modeling_marian import (
    MarianAttention,
    MarianDecoder,
    MarianEncoderLayer,
    MarianMTModel,
    MarianModel,
    MarianPreTrainedModel,
    eager_attention_forward,
)


class MarianRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(
        self,
        config: MarianConfig,
        device: torch.device,
        is_decoder: bool = True,
        rope_theta: float = 10000.0,
    ):
        super().__init__()

        self.config = config
        self.is_decoder = is_decoder
        self.rope_theta = rope_theta

        hidden_size = getattr(config, "hidden_size", None) or config.d_model
        num_heads = (
            getattr(config, "num_attention_heads", None)
            or (config.decoder_attention_heads if is_decoder else config.encoder_attention_heads)
        )

        self.hidden_size = hidden_size
        self.num_attention_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        inv_freq, attention_scaling = self.compute_default_rope_parameters(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            rope_theta=self.rope_theta,
            device=torch.device("cpu"),
        )

        self.attention_scaling = attention_scaling
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = inv_freq

    @staticmethod
    def compute_default_rope_parameters(
        hidden_size: int,
        num_attention_heads: int,
        device: torch.device,
        rope_theta: float = 10000.0,
    ) -> tuple[torch.Tensor, float]:
        dim = hidden_size // num_attention_heads
        base = rope_theta
        attention_factor = 1.0
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryMarianAttention(MarianAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        is_cross_attention = key_value_states is not None

        bsz, tgt_len = hidden_states.shape[:-1]
        src_len = key_value_states.shape[1] if is_cross_attention else tgt_len

        q_input_shape = (bsz, tgt_len, -1, self.head_dim)
        kv_input_shape = (bsz, src_len, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(*q_input_shape).transpose(1, 2)

        is_updated = False
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    curr_past_key_values = past_key_values.cross_attention_cache
                else:
                    curr_past_key_values = past_key_values.self_attention_cache
            else:
                curr_past_key_values = past_key_values

        current_states = key_value_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_values is not None and is_updated:
            key_states = curr_past_key_values.layers[self.layer_idx].keys
            value_states = curr_past_key_values.layers[self.layer_idx].values
        else:
            key_states = self.k_proj(current_states)
            value_states = self.v_proj(current_states)
            key_states = key_states.view(*kv_input_shape).transpose(1, 2)
            value_states = value_states.view(*kv_input_shape).transpose(1, 2)

            # Apply RoPE for self-attention only (not cross-attention)
            if not is_cross_attention:
                assert position_embeddings, "no position embeddings provided to self attention layer."
                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_values is not None:
                cache_position = cache_position if not is_cross_attention else None
                if not is_cross_attention:
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                else:
                    cache_kwargs = {"cache_position": cache_position}
                key_states, value_states = curr_past_key_values.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
                if is_cross_attention and isinstance(past_key_values, EncoderDecoderCache):
                    past_key_values.is_updated[self.layer_idx] = True

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scaling,
            output_attentions=output_attentions,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights


class RotaryMarianEncoderLayer(MarianEncoderLayer):
    def __init__(self, config: MarianConfig, layer_idx: Optional[int] = None):
        MarianEncoderLayer.__init__(self, config, layer_idx)
        self.embed_dim = config.d_model

        self.self_attn = RotaryMarianAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
            layer_idx=layer_idx,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        position_embeddings: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        output_attentions: Optional[bool] = False,
    ) -> tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        residual = hidden_states
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class SemiRotaryMarianEncoder(MarianPreTrainedModel):
    def __init__(self, config: MarianConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        # RoPE for encoder positions
        self.rotary_emb = MarianRotaryEmbedding(config=config, device=self.device, is_decoder=False)
        self.layers = nn.ModuleList([RotaryMarianEncoderLayer(config, layer_idx=i) for i in range(config.encoder_layers)])

        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple[torch.Tensor], BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # Build RoPE position ids and embeddings
        batch_size, seq_length = inputs_embeds.size()[:-1]
        device = inputs_embeds.device
        position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = nn.functional.dropout(inputs_embeds, p=self.dropout, training=self.training)

        attention_mask = create_bidirectional_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    to_drop = True

            if to_drop:
                layer_outputs = (None, None)
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    output_attentions=output_attentions,
                )
                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class SemiRotaryMarianModel(MarianModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: MarianConfig):
        MarianModel.__init__(self, config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size

        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        if self.config.share_encoder_decoder_embeddings:
            encoder_embed_tokens = decoder_embed_tokens = self.shared
        else:
            encoder_embed_tokens = copy.deepcopy(self.shared)
            decoder_embed_tokens = copy.deepcopy(self.shared)
            self.shared = None

        self.encoder = SemiRotaryMarianEncoder(config, encoder_embed_tokens)
        self.decoder = MarianDecoder(config)
        if decoder_embed_tokens is not None:
            self.decoder.embed_tokens = decoder_embed_tokens

        self.post_init()


class SemiRotaryMarianMTModel(MarianMTModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        "final_logits_bias",
        "encoder.embed_positions.weight",
    ]
    _keys_to_ignore_on_save = ["model.encoder.embed_positions.weight"]
    _tied_weights_keys = ["model.encoder.embed_tokens.weight", "model.decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: MarianConfig):
        MarianMTModel.__init__(self, config)
        self.model = SemiRotaryMarianModel(config)

        target_vocab_size = config.vocab_size if config.share_encoder_decoder_embeddings else config.decoder_vocab_size
        self.register_buffer("final_logits_bias", torch.zeros((1, target_vocab_size)))
        self.lm_head = nn.Linear(config.d_model, target_vocab_size, bias=False)

        self.post_init()


__all__ = ["SemiRotaryMarianModel", "SemiRotaryMarianMTModel", "MarianPreTrainedModel"]


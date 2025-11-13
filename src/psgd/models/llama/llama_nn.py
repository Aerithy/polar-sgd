# llama_nn.py
from dataclasses import dataclass
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class LlamaConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    rope_theta: float = 10000.0
    pad_token_id: int = 0
    tie_word_embeddings: bool = True  # Will be handled carefully in PP


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, rope_theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = rope_theta

    def _build_freqs(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        half_dim = self.dim // 2
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, half_dim, device=device, dtype=torch.float32) / half_dim))
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = t[:, None] * inv_freq[None, :]
        cos = torch.cos(freqs).to(dtype=dtype)
        sin = torch.sin(freqs).to(dtype=dtype)
        return cos, sin

    def apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, H, T, D = x.shape
        x_ = x.view(B, H, T, D // 2, 2)
        x1, x2 = x_[..., 0], x_[..., 1]
        cos = cos[None, None, :, :]  # [1, 1, T, D//2]
        sin = sin[None, None, :, :]
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        return torch.stack([x1_rot, x2_rot], dim=-1).reshape(B, H, T, D)


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        assert self.hidden_size % self.num_heads == 0
        self.head_dim = self.hidden_size // self.num_heads

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, config.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        B, T, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.rotary.apply_rotary(q, cos, sin)
        k = self.rotary.apply_rotary(k, cos, sin)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        causal_mask = torch.full((T, T), float("-inf"), device=x.device, dtype=attn_scores.dtype).triu(1)
        attn_scores = attn_scores + causal_mask

        # Optional padding mask (assumed consistent across stages)
        if attention_mask is not None:
            mask = (1.0 - attention_mask.float()) * torch.finfo(x.dtype).min
            attn_scores = attn_scores + mask[:, None, None, :]

        attn = torch.softmax(attn_scores, dim=-1).to(v.dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)
        self.attn = LlamaAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size)
        self.mlp = LlamaMLP(config)
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rotary = RotaryEmbedding(self.head_dim, config.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        print(f"[LlamaDecoderLayer.forward] input shape: {x.shape}, dtype: {x.dtype}")
        seq_len = x.shape[1]
        cos, sin = self.rotary._build_freqs(seq_len, x.device, x.dtype)
        x = x + self.attn(self.attn_norm(x), cos, sin, attention_mask)
        x = x + self.mlp(self.mlp_norm(x))
        print(f"[LlamaDecoderLayer.forward] output shape: {x.shape}, dtype: {x.dtype}")
        return x


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleDict({
            str(i): LlamaDecoderLayer(config) for i in range(config.num_hidden_layers)
        })
        self.final_norm = RMSNorm(config.hidden_size)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids_or_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        print(f"[LlamaModel.forward] input shape: {input_ids_or_hidden.shape}, dtype: {input_ids_or_hidden.dtype}")
        
        if self.embed_tokens is not None:
            # Stage 0: input is token IDs
            assert input_ids_or_hidden.dim() == 2
            hidden_states = self.embed_tokens(input_ids_or_hidden)
        else:
            # Stage 1+: input is hidden states
            assert input_ids_or_hidden.dim() == 3
            hidden_states = input_ids_or_hidden

        layer_keys = sorted(int(k) for k in self.layers.keys())
        for layer_id in layer_keys:
            layer = self.layers[str(layer_id)]
            print(f"[LlamaModel] before layer {layer_id}: shape={hidden_states.shape}, dtype={hidden_states.dtype}")
            hidden_states = layer(hidden_states, attention_mask)
            print(f"[LlamaModel] after layer {layer_id}: shape={hidden_states.shape}, dtype={hidden_states.dtype}")

        if self.final_norm is not None:
            hidden_states = self.final_norm(hidden_states)

        print(f"[LlamaModel.forward] output shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")
        return hidden_states


class MyLlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # ⚠️ DO NOT tie weights here if using pipeline parallelism!
        # Weight tying will be handled during checkpoint loading if needed.
        # if config.tie_word_embeddings:
        #     self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids_or_hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        # labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden = self.model(input_ids_or_hidden, attention_mask)
        if self.lm_head is not None:
            return self.lm_head(hidden)
        else:
            return hidden  # hidden states for intermediate stages
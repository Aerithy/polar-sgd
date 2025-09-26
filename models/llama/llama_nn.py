# classes: LlamaConfig, RMSNorm, RotaryEmbedding, LlamaAttention, LlamaMLP, LlamaDecoderLayer, LlamaModel, CausalLMOutput, MyLlamaForCausalLM
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
    tie_word_embeddings: bool = True

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, rope_theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = rope_theta

    def _build_freqs(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        # angles shape: [seq_len, dim//2]
        half_dim = self.dim // 2
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, half_dim, device=device, dtype=torch.float32) / half_dim))
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        # freqs = torch.einsum("i,j->ij", t, inv_freq)  # [seq_len, half_dim]
        freqs = t[:, None] * inv_freq[None, :]
        cos = torch.cos(freqs).to(dtype=dtype)
        sin = torch.sin(freqs).to(dtype=dtype)
        return cos, sin

    def apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: [B, H, T, D], cos/sin: [T, D/2]
        B, H, T, D = x.shape
        x_ = x.view(B, H, T, D // 2, 2)
        x1 = x_[..., 0]
        x2 = x_[..., 1]
        # 修复：cos/sin 应该被扩展为 4D 张量 [1, 1, T, D/2] 以匹配 x1/x2 的广播。
        # 错误日志表明这里可能被错误地实现为了5D张量。
        cos = cos[None, None, :, :]  # Shape: [1, 1, T, D/2]
        sin = sin[None, None, :, :]  # Shape: [1, 1, T, D/2]
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        out = torch.stack([x1_rot, x2_rot], dim=-1).reshape(B, H, T, D)
        return out

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

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # x: [B, T, C], attention_mask: [B, T] with 1 for valid, 0 for pad
        B, T, C = x.shape
        q = self.q_proj(x)  # [B, T, C]
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # cos, sin = self.rotary._build_freqs(T, x.device, x.dtype)
        q = self.rotary.apply_rotary(q, cos, sin)
        k = self.rotary.apply_rotary(k, cos, sin)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, T, T]

        # Causal mask
        causal_mask = torch.full((T, T), float("-inf"), device=x.device, dtype=attn_scores.dtype).triu(1)
        attn_scores = attn_scores + causal_mask

        # Key padding mask: attention_mask [B, T], 1 for valid, 0 for pad
        if attention_mask is not None:
            # convert to [B, 1, 1, T] additive mask with -inf on pads
            mask = (1.0 - attention_mask.float()) * -1e10
            attn_scores = attn_scores + mask[:, None, None, :].to(attn_scores.device)

        attn = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn, v)  # [B, H, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]
        out = self.o_proj(out)
        return out

class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (SiLU(gate) * up) -> down
        gated = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gated * up)

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)
        self.attn = LlamaAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size)
        self.mlp = LlamaMLP(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Pre-norm residual
        h = x + self.attn(self.attn_norm(x), cos, sin, attention_mask=attention_mask)
        h = h + self.mlp(self.mlp_norm(h))
        return h

class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.final_norm = RMSNorm(config.hidden_size)
        
        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_emb = RotaryEmbedding(head_dim, config.rope_theta)

        self._init_weights()

    def _init_weights(self):
        # Simple init; for production, consider scaled init per LLaMA
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # input_ids: [B, T], attention_mask: [B, T]
        x = self.embed_tokens(input_ids)  # [B, T, C]
        
        seq_len = input_ids.shape[1]
        cos, sin = self.rotary_emb._build_freqs(seq_len, x.device, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin, attention_mask=attention_mask)
        x = self.final_norm(x)
        return x  # [B, T, C]

@dataclass
class CausalLMOutput:
    loss: Optional[torch.Tensor]
    logits: torch.Tensor

class MyLlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            # weight tying
            self.lm_head.weight = self.model.embed_tokens.weight
        self.export_mode: bool = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> CausalLMOutput:
        hidden_states = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)  # [B, T, V]

        try:
            import torch._dynamo as _dynamo
            if getattr(self, "export_mode", False) or (_dynamo.is_compiling()):
                return logits  # 仅返回 Tensor 以便 export
        except Exception:
            pass
        
        loss = None
        if labels is not None:
            # Shift-one token LM loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )
        return CausalLMOutput(loss=loss, logits=logits)
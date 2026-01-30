"""
model/model.py — ~24M parameter language model (the model's architecture)
All components hand-implemented: RMSNorm, RoPE, SwiGLU, MHA, KVCache, TransformerBlock
No transformers, no einops, no flash-attn. Pure torch.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


@dataclass
class SanaConfig:
    vocab_size: int = 16000
    hidden_dim: int = 384
    num_layers: int = 10
    num_heads: int = 6
    ffn_multiplier: float = 2.667
    max_seq_len: int = 512
    dropout: float = 0.1
    rope_base: int = 10000
    tie_embeddings: bool = True

    def __post_init__(self):
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
            )

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_heads

    @property
    def ffn_intermediate(self) -> int:
        return round(self.hidden_dim * self.ffn_multiplier)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SanaConfig":
        with open(path) as f:
            d = json.load(f)
        return cls(**d)


@dataclass
class KVCache:
    k: torch.Tensor
    v: torch.Tensor


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: int = 10000):
        super().__init__()
        self.head_dim = head_dim

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

        t = torch.arange(max_seq_len).float()

        freqs = torch.outer(t, inv_freq)

        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        T = x.shape[2]
        cos = self.cos_cached[start_pos : start_pos + T]
        sin = self.sin_cached[start_pos : start_pos + T]

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        rot_x1 = x1 * cos - x2 * sin
        rot_x2 = x1 * sin + x2 * cos

        out = torch.stack([rot_x1, rot_x2], dim=-1).flatten(-2)
        return out


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class Attention(nn.Module):
    def __init__(self, config: SanaConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_dim = config.hidden_dim
        self.max_seq_len = config.max_seq_len

        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        self.rope = RotaryEmbedding(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_base,
        )

        mask = torch.full((config.max_seq_len, config.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer(
            "causal_mask", mask.unsqueeze(0).unsqueeze(0), persistent=False
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, _ = x.shape
        H = self.num_heads
        D = self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        q = self.rope(q, start_pos=start_pos)
        k = self.rope(k, start_pos=start_pos)

        if kv_cache is not None:
            k = torch.cat([kv_cache.k, k], dim=2)
            v = torch.cat([kv_cache.v, v], dim=2)

            if k.shape[2] > self.max_seq_len:
                k = k[:, :, -self.max_seq_len :, :]
                v = v[:, :, -self.max_seq_len :, :]
            new_cache = KVCache(k=k, v=v)
        else:
            new_cache = None

        T_k = k.shape[2]

        scale = math.sqrt(D)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if kv_cache is None:
            scores = scores + self.causal_mask[:, :, :T, :T_k]
        else:

            if T > 1:
                scores = scores + self.causal_mask[:, :, :T, :T_k]

        attn_weights = F.softmax(scores.float(), dim=-1).to(x.dtype)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.hidden_dim)
        out = self.out_proj(out)

        return out, new_cache


class TransformerBlock(nn.Module):
    def __init__(self, config: SanaConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_dim)
        self.ffn_norm = RMSNorm(config.hidden_dim)
        self.attn = Attention(config)
        self.ffn = SwiGLUFFN(config.hidden_dim, config.ffn_intermediate)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:

        attn_out, new_cache = self.attn(
            self.attn_norm(x), kv_cache=kv_cache, start_pos=start_pos
        )
        x = x + attn_out

        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


class SanaModel(nn.Module):
    def __init__(self, config: SanaConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )

        self.final_norm = RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        self.use_gradient_checkpointing = False
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small normal distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.shape

        x = self.dropout(self.embedding(input_ids))

        for layer in self.layers:
            if self.use_gradient_checkpointing and self.training:

                x, _ = grad_checkpoint(
                    lambda h, l=layer: l(h, kv_cache=None, start_pos=0),
                    x,
                    use_reentrant=False,
                )
            else:
                x, _ = layer(x, kv_cache=None, start_pos=0)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:

            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        eos_token_id: int = 3,
        repetition_penalty: float = 1.3,
    ) -> torch.Tensor:
        """
        Autoregressive generation with KV cache and nucleus sampling.
        Returns generated token ids (including prompt).
        """
        device = input_ids.device
        B, T_prompt = input_ids.shape
        assert B == 1, "generate() only supports batch_size=1"

        kv_caches: List[Optional[KVCache]] = [
            KVCache(
                k=torch.zeros(
                    1,
                    self.config.num_heads,
                    0,
                    self.config.hidden_dim // self.config.num_heads,
                    device=input_ids.device,
                ),
                v=torch.zeros(
                    1,
                    self.config.num_heads,
                    0,
                    self.config.hidden_dim // self.config.num_heads,
                    device=input_ids.device,
                ),
            )
            for _ in self.layers
        ]

        x = self.dropout(self.embedding(input_ids))
        for i, layer in enumerate(self.layers):
            x, kv_caches[i] = layer(x, kv_cache=kv_caches[i], start_pos=0)
        x = self.final_norm(x)

        generated_ids = input_ids[0].tolist()
        current_len = T_prompt

        for _ in range(max_new_tokens):

            logits = self.lm_head(x[:, -1, :])
            logits = logits[0]

            if repetition_penalty != 1.0:
                for token_id in set(generated_ids):
                    if logits[token_id] > 0:
                        logits[token_id] /= repetition_penalty
                    else:
                        logits[token_id] *= repetition_penalty

            if temperature != 1.0:
                logits = logits / temperature

            probs = F.softmax(logits.float(), dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)

            sorted_probs[cumulative - sorted_probs > top_p] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum()

            sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_idx[sampled_idx].item()

            generated_ids.append(next_token)

            if next_token == eos_token_id:
                break

            next_token_tensor = torch.tensor(
                [[next_token]], dtype=torch.long, device=device
            )
            x = self.dropout(self.embedding(next_token_tensor))

            start_pos = current_len
            current_len += 1

            for i, layer in enumerate(self.layers):
                x, kv_caches[i] = layer(x, kv_cache=kv_caches[i], start_pos=start_pos)

            x = self.final_norm(x)

        return torch.tensor(generated_ids, dtype=torch.long, device=device).unsqueeze(0)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def count_params(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total/1e6:.2f}M | Trainable: {trainable/1e6:.2f}M"


if __name__ == "__main__":
    cfg = SanaConfig()
    model = SanaModel(cfg)
    print("Sana 🪼 model initialized")
    print(count_params(model))

    x = torch.randint(0, cfg.vocab_size, (2, 64))
    labels = x.clone()
    logits, loss = model(x, labels)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

    prompt = torch.randint(0, cfg.vocab_size, (1, 8))
    out = model.generate(prompt, max_new_tokens=20)
    print(f"Generated shape: {out.shape}")
    print("Model self-test passed ✓")

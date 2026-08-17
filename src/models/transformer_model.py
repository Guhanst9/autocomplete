import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


Cache = tuple[torch.Tensor, torch.Tensor]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError("attention head dimension must be even")
        inverse_frequency = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(
            position_offset,
            position_offset + query.size(-2),
            device=query.device,
            dtype=torch.float32,
        )
        angles = torch.outer(positions, self.inverse_frequency.float())
        angles = torch.cat((angles, angles), dim=-1)
        cosine = angles.cos().to(query.dtype)[None, None, :, :]
        sine = angles.sin().to(query.dtype)[None, None, :, :]
        return (
            query * cosine + _rotate_half(query) * sine,
            key * cosine + _rotate_half(key) * sine,
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, Optional[Cache]]:
        batch_size, sequence_length, d_model = x.shape
        query, key, value = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(
                batch_size,
                sequence_length,
                self.n_heads,
                self.head_dim,
            ).transpose(1, 2)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)
        query, key = self.rotary(query, key, position_offset)

        if past_key_value is not None:
            if sequence_length != 1:
                raise ValueError("cached attention accepts one new token at a time")
            past_key, past_value = past_key_value
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        dropout = self.dropout if self.training else 0.0
        if past_key_value is not None:
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=dropout,
                is_causal=False,
            )
        else:
            # dna batches are right-padded, so valid causal positions never attend to padding
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=dropout,
                is_causal=True,
            )

        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            d_model,
        )
        cache = (key, value) if use_cache else None
        return self.out(attended), cache


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads, dropout)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, Optional[Cache]]:
        attended, cache = self.attention(
            self.attention_norm(x),
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            position_offset=position_offset,
        )
        x = x + self.residual_dropout(attended)
        x = x + self.mlp(self.mlp_norm(x))
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        return x, cache


class TransformerSequenceModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 9,
        ffn_dim: int = 1600,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        mask_token_id: int = 1,
        eos_token_id: Optional[int] = None,
        max_length: int = 1024,
        input_vocab_size: Optional[int] = None,
        output_vocab_size: Optional[int] = None,
    ):
        super().__init__()
        self.input_vocab_size = input_vocab_size or vocab_size
        self.output_vocab_size = output_vocab_size or vocab_size
        self.vocab_size = self.output_vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.eos_token_id = eos_token_id
        self.max_length = max_length
        self.model_type = "transformer"

        self.embed = nn.Embedding(self.input_vocab_size, d_model, padding_idx=pad_token_id)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, ffn_dim, dropout)
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, self.output_vocab_size, bias=False)
        self.apply(_init_weights)
        if self.input_vocab_size == self.output_vocab_size:
            self.lm_head.weight = self.embed.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[Cache]] = None,
        use_cache: bool = False,
        position_offset: int = 0,
    ):
        if input_ids.size(1) > self.max_length:
            raise ValueError(
                f"input length {input_ids.size(1)} exceeds context length {self.max_length}"
            )
        if past_key_values is not None and len(past_key_values) != self.n_layers:
            raise ValueError("past_key_values must contain one cache per layer")

        x = self.embedding_dropout(self.embed(input_ids))
        new_cache = []
        for index, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[index]
            x, layer_cache = block(
                x,
                attention_mask=attention_mask,
                past_key_value=past,
                use_cache=use_cache,
                position_offset=position_offset,
            )
            if use_cache:
                new_cache.append(layer_cache)
        logits = self.lm_head(self.ln_f(x))
        return (logits, new_cache) if use_cache else logits

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
        objective: str = "autocomplete",
        eos_loss_weight: float = 1.0,
        logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if logits is None:
            logits = self.forward(input_ids, attention_mask=attention_mask)
        labels = target_ids.clone()
        labels[attention_mask == 0] = -100
        if objective == "masked":
            labels[input_ids != self.mask_token_id] = -100
        elif objective == "autocomplete":
            if loss_mask is None:
                loss_mask = (target_ids != self.pad_token_id) & attention_mask.bool()
            labels[loss_mask == 0] = -100
        else:
            raise ValueError("objective must be 'masked' or 'autocomplete'")

        class_weights = None
        if eos_loss_weight != 1.0 and self.eos_token_id is not None:
            class_weights = torch.ones(
                self.vocab_size,
                dtype=logits.dtype,
                device=logits.device,
            )
            class_weights[self.eos_token_id] = eos_loss_weight
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
            weight=class_weights,
        )

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        use_recurrent: bool = True,
        eos_token_id: Optional[int] = None,
        stop_at_eos: bool = True,
        forbidden_token_ids: Optional[tuple[int, ...]] = None,
        min_new_tokens: int = 0,
        sampling_temperature: Optional[float] = None,
    ) -> torch.Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")
        if sampling_temperature is not None and sampling_temperature <= 0:
            raise ValueError("sampling_temperature must be positive")
        if prompt_ids.size(1) == 0:
            raise ValueError("prompt cannot be empty")
        if prompt_ids.size(1) > self.max_length:
            warnings.warn(
                f"prompt exceeds {self.max_length} bases; using the most recent context",
                stacklevel=2,
            )
        if not use_recurrent:
            return self._generate_without_cache(
                prompt_ids,
                max_new_tokens,
                eos_token_id,
                stop_at_eos,
                forbidden_token_ids,
                min_new_tokens,
                sampling_temperature,
            )

        eos_token_id = self.eos_token_id if eos_token_id is None else eos_token_id
        batch_size, prompt_length = prompt_ids.shape
        context = prompt_ids[:, -self.max_length :]
        context_start = prompt_length - context.size(1)
        logits, cache = self.forward(
            context,
            use_cache=True,
            position_offset=context_start,
        )
        next_logits = logits[:, -1, :]
        generated = prompt_ids.tolist()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=prompt_ids.device)

        for generated_count in range(max_new_tokens):
            next_token = _choose_token(
                next_logits,
                forbidden_token_ids,
                eos_token_id,
                stop_at_eos,
                min_new_tokens,
                generated_count,
                sampling_temperature,
                finished,
            )
            for batch_index in range(batch_size):
                generated[batch_index].append(next_token[batch_index].item())
            if stop_at_eos and eos_token_id is not None:
                finished = finished | (next_token == eos_token_id)
                if finished.all().item():
                    break
            if generated_count + 1 == max_new_tokens:
                break

            cache = [
                (key[:, :, -(self.max_length - 1) :, :], value[:, :, -(self.max_length - 1) :, :])
                for key, value in cache
            ]
            next_position = prompt_length + generated_count
            step_logits, cache = self.forward(
                next_token[:, None],
                past_key_values=cache,
                use_cache=True,
                position_offset=next_position,
            )
            next_logits = step_logits[:, -1, :]

        return torch.tensor(generated, dtype=prompt_ids.dtype, device=prompt_ids.device)

    @torch.no_grad()
    def generate_triplets(
        self,
        prompt_ids: torch.Tensor,
        triplet_base_ids: torch.Tensor,
        max_new_bases: int,
        sampling_temperature: Optional[float] = None,
    ) -> torch.Tensor:
        if max_new_bases < 0:
            raise ValueError("max_new_bases cannot be negative")
        if prompt_ids.size(1) == 0:
            raise ValueError("prompt cannot be empty")
        batch, prompt_length = prompt_ids.shape
        context = prompt_ids[:, -self.max_length :]
        context_start = prompt_length - context.size(1)
        logits, cache = self.forward(context, use_cache=True, position_offset=context_start)
        next_logits = logits[:, -1, :]
        generated = prompt_ids.tolist()
        generated_count = 0
        triplet_base_ids = triplet_base_ids.to(prompt_ids.device)

        while generated_count < max_new_bases:
            class_ids = _choose_token(
                next_logits,
                None,
                None,
                False,
                0,
                generated_count,
                sampling_temperature,
                torch.zeros(batch, dtype=torch.bool, device=prompt_ids.device),
            )
            next_bases = triplet_base_ids[class_ids]
            for offset in range(3):
                if generated_count >= max_new_bases:
                    break
                next_token = next_bases[:, offset]
                for batch_index in range(batch):
                    generated[batch_index].append(next_token[batch_index].item())
                generated_count += 1
                if generated_count >= max_new_bases:
                    break
                if self.max_length > 1:
                    cache = [
                        (
                            key[:, :, -(self.max_length - 1) :, :],
                            value[:, :, -(self.max_length - 1) :, :],
                        )
                        for key, value in cache
                    ]
                position = prompt_length + generated_count - 1
                step_logits, cache = self.forward(
                    next_token[:, None],
                    past_key_values=cache,
                    use_cache=True,
                    position_offset=position,
                )
                next_logits = step_logits[:, -1, :]
        return torch.tensor(generated, dtype=prompt_ids.dtype, device=prompt_ids.device)

    @torch.no_grad()
    def _generate_without_cache(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: Optional[int],
        stop_at_eos: bool,
        forbidden_token_ids: Optional[tuple[int, ...]],
        min_new_tokens: int,
        sampling_temperature: Optional[float],
    ) -> torch.Tensor:
        eos_token_id = self.eos_token_id if eos_token_id is None else eos_token_id
        generated = prompt_ids.tolist()
        finished = torch.zeros(prompt_ids.size(0), dtype=torch.bool, device=prompt_ids.device)
        for generated_count in range(max_new_tokens):
            total_length = len(generated[0])
            context_start = max(0, total_length - self.max_length)
            context = torch.tensor(
                [tokens[context_start:] for tokens in generated],
                dtype=prompt_ids.dtype,
                device=prompt_ids.device,
            )
            logits = self.forward(context, position_offset=context_start)[:, -1, :]
            next_token = _choose_token(
                logits,
                forbidden_token_ids,
                eos_token_id,
                stop_at_eos,
                min_new_tokens,
                generated_count,
                sampling_temperature,
                finished,
            )
            for batch_index in range(prompt_ids.size(0)):
                generated[batch_index].append(next_token[batch_index].item())
            if stop_at_eos and eos_token_id is not None:
                finished = finished | (next_token == eos_token_id)
                if finished.all().item():
                    break
        return torch.tensor(generated, dtype=prompt_ids.dtype, device=prompt_ids.device)


def _choose_token(
    logits: torch.Tensor,
    forbidden_token_ids: Optional[tuple[int, ...]],
    eos_token_id: Optional[int],
    stop_at_eos: bool,
    min_new_tokens: int,
    generated_count: int,
    sampling_temperature: Optional[float],
    finished: torch.Tensor,
) -> torch.Tensor:
    logits = logits.clone()
    if forbidden_token_ids:
        valid = [token for token in forbidden_token_ids if 0 <= token < logits.size(-1)]
        logits[:, valid] = -1e10
    if (
        stop_at_eos
        and eos_token_id is not None
        and generated_count < min_new_tokens
        and 0 <= eos_token_id < logits.size(-1)
    ):
        logits[:, eos_token_id] = -1e10
    valid_rows = torch.isfinite(logits).any(dim=-1) & (logits > -1e9).any(dim=-1)
    if not valid_rows.all().item():
        raise RuntimeError("generation has no valid next token")
    if sampling_temperature is None:
        next_token = logits.argmax(dim=-1)
    else:
        probabilities = torch.softmax(logits / sampling_temperature, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
    if stop_at_eos and eos_token_id is not None:
        eos = torch.full_like(next_token, eos_token_id)
        next_token = torch.where(finished, eos, next_token)
    return next_token


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)

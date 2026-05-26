"""
Full S4 model for protein sequence autocompletion.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    from src.models.s4.s4_layer import S4Block
except ImportError:
    from .s4.s4_layer import S4Block


def _get_config(size: str) -> dict:
    if size == "small":
        return dict(n_layers=4, d_model=128, d_state=64)
    if size == "base":
        return dict(n_layers=6, d_model=256, d_state=64)
    if size == "large":
        return dict(n_layers=12, d_model=512, d_state=64)
    return {}


class S4ProteinModel(nn.Module):
    """
    S4 model for protein sequence autocompletion:
    embed -> N x S4Block -> ln -> output projection -> logits.
    """

    def __init__(
        self,
        vocab_size: int = 23,
        d_model: int = 256,
        d_state: int = 64,
        n_layers: int = 6,
        dropout: float = 0.1,
        kernel_type: str = "diag",
        bidirectional: bool = True,
        l_max: Optional[int] = None,
        pad_token_id: int = 0,
        max_length: int = 1024,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.mask_token_id = 1
        self.max_length = max_length

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.blocks = nn.ModuleList([
            S4Block(
                d_model=d_model,
                d_state=d_state,
                dropout=dropout,
                kernel_type=kernel_type,
                bidirectional=bidirectional,
                l_max=l_max,
            )
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.embed.weight.data.normal_(mean=0.0, std=0.02)
        self.apply(_init_weights)


    # embedding -> s4 blocks -> layer norm -> output projection pipeline
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embed(input_ids)  # tokens -> vectors
        for block in self.blocks:  # n × s4 block processing
            x, _ = block(x, attention_mask=attention_mask)
        x = self.ln_f(x)  # layer normalization
        logits = self.lm_head(x)  # -> amino acid logits
        return logits



    def compute_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy on masked positions only."""
        logits = self.forward(input_ids, attention_mask=attention_mask)
        labels = target_ids.clone()
        labels[attention_mask == 0] = -100
        labels[input_ids != self.mask_token_id] = -100
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, self.vocab_size),
            labels.view(-1),
            reduction="mean",
            ignore_index=-100,
        )
        return loss

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        use_recurrent: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressive generation using recurrent step mode (O(1) per token).
        
        Args:
            prompt_ids: (batch, prompt_len) - input sequence
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: top-k filtering
            top_p: nucleus sampling threshold
            do_sample: if False, use greedy decoding
            use_recurrent: if True, use O(1) recurrent step mode (faster for long generation)
        
        Returns: (batch, prompt_len + max_new_tokens)
        """
        batch, prompt_len = prompt_ids.shape
        device = prompt_ids.device
        
        if use_recurrent:
            return self._generate_recurrent(
                prompt_ids, max_new_tokens, temperature, top_k, top_p, do_sample
            )
        else:
            return self._generate_forward(
                prompt_ids, max_new_tokens, temperature, top_k, top_p, do_sample
            )
    
    @torch.no_grad()
    def _generate_recurrent(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        do_sample: bool,
    ) -> torch.Tensor:
        """
        Recurrent step-based generation - O(1) per token after prompt processing.
        Uses S4's recurrent mode for efficient autoregressive inference.
        """
        batch, prompt_len = prompt_ids.shape
        device = prompt_ids.device
        
        # Initialize states for each layer
        states = [block.default_state(batch, device) for block in self.blocks]
        
        # Process prompt tokens to build up state
        for t in range(prompt_len):
            token = prompt_ids[:, t]  # (batch,)
            x = self.embed(token)     # (batch, d_model)
            
            # Step through each block, updating states
            for i, block in enumerate(self.blocks):
                x, states[i] = block.step(x, states[i])
        
        # Generate new tokens using recurrent step mode
        generated = prompt_ids.tolist()
        
        for _ in range(max_new_tokens):
            # Get logits from current hidden state
            x_final = self.ln_f(x)
            logits = self.lm_head(x_final)  # (batch, vocab_size)
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = -1e10
            
            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                sorted_logits[remove] = -1e10
                logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)
            
            # Sample or greedy
            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, 1).squeeze(-1)
            else:
                next_tok = logits.argmax(dim=-1)
            
            # Append to generated
            for i in range(batch):
                generated[i].append(next_tok[i].item())
            
            # Step the model with new token (O(1) operation!)
            x = self.embed(next_tok)  # (batch, d_model)
            for i, block in enumerate(self.blocks):
                x, states[i] = block.step(x, states[i])
        
        return torch.tensor(generated, device=device, dtype=prompt_ids.dtype)
    
    @torch.no_grad()
    def _generate_forward(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        do_sample: bool,
    ) -> torch.Tensor:
        """
        Forward-based generation - re-encodes entire sequence each step.
        Slower but uses convolution mode (good for short sequences).
        """
        batch, prompt_len = prompt_ids.shape
        device = prompt_ids.device
        generated = list(prompt_ids.cpu().tolist())
        
        for _ in range(max_new_tokens):
            context = torch.tensor([s[-self.max_length:] for s in generated], device=device)
            logits = self.forward(context)[:, -1, :]
            
            if temperature != 1.0:
                logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = -1e10
            if top_p is not None:
                sorted_logits, _ = torch.sort(logits, descending=True)
                cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                logits[logits < sorted_logits[remove].min(dim=-1, keepdim=True).values] = -1e10
            
            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, 1).squeeze(-1)
            else:
                next_tok = logits.argmax(dim=-1)
            
            for i in range(batch):
                generated[i].append(next_tok[i].item())
        
        return torch.tensor(generated, device=device, dtype=prompt_ids.dtype)

    @classmethod
    def from_preset(cls, size: str, **kwargs) -> "S4ProteinModel":
        cfg = _get_config(size)
        cfg.update(kwargs)
        return cls(**cfg)


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

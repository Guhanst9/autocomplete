"""
Variant effect prediction: delta log-likelihood between wild-type and mutant sequence.
"""
import re
from typing import Optional

import torch

try:
    from src.models.s4_model import S4ProteinModel
    from src.dataloaders.protein import ProteinTokenizer
except ImportError:
    from models.s4_model import S4ProteinModel
    from dataloaders.protein import ProteinTokenizer


class VariantEffectPredictor:
    """
    Given wild-type sequence and mutation (e.g. "A123G"), compute
    delta log-likelihood (mutant - wild-type) as effect score.
    """

    def __init__(self, model: S4ProteinModel, tokenizer: Optional[ProteinTokenizer] = None):
        self.model = model
        self.tokenizer = tokenizer or ProteinTokenizer()
        self.model.eval()

    def _log_likelihood(self, seq_ids: list, device: torch.device) -> torch.Tensor:
        """Sum of log probs for each position given previous context."""
        if len(seq_ids) <= 1:
            return torch.tensor(0.0, device=device)
        input_ids = torch.tensor([seq_ids[:-1]], dtype=torch.long, device=device)
        target_ids = torch.tensor([seq_ids[1:]], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids, device=device)
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
        log_probs = torch.log_softmax(logits, dim=-1)
        nll = torch.nn.functional.nll_loss(
            log_probs.view(-1, log_probs.size(-1)),
            target_ids.view(-1),
            reduction="sum",
        )
        return -nll

    def predict_variant_effect(
        self,
        wt_sequence: str,
        mutation: str,
        device: Optional[torch.device] = None,
    ) -> float:
        """
        mutation: e.g. "A123G" = position 123 (1-based) wild-type A -> G.
        Returns delta log-likelihood (mutant - wt). Positive = mutant more likely.
        """
        if device is None:
            device = next(self.model.parameters()).device
        m = re.match(r"([A-Z])(\d+)([A-Z])", mutation.upper())
        if not m:
            raise ValueError("mutation must be like A123G (wt, position 1-based, mutant)")
        wt_aa, pos_str, mut_aa = m.groups()
        pos = int(pos_str)
        if pos < 1:
            raise ValueError("position must be 1-based")
        wt_ids = self.tokenizer.encode(wt_sequence)
        if pos > len(wt_ids):
            raise ValueError("position beyond sequence length")
        idx_wt = self.tokenizer.vocab.get(wt_aa, self.tokenizer.vocab[self.tokenizer.unk_token])
        idx_mut = self.tokenizer.vocab.get(mut_aa, self.tokenizer.vocab[self.tokenizer.unk_token])
        if wt_ids[pos - 1] != idx_wt:
            raise ValueError(f"wild-type residue at position {pos} is not {wt_aa}")
        mut_ids = wt_ids.copy()
        mut_ids[pos - 1] = idx_mut
        ll_wt = self._log_likelihood(wt_ids, device).item()
        ll_mut = self._log_likelihood(mut_ids, device).item()
        return ll_mut - ll_wt

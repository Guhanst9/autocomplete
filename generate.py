import argparse
import os
from typing import Optional

import torch

try:
    from src.models.s4_model import S4ProteinModel, adapt_state_dict_vocab
    from src.dataloaders.protein import ProteinTokenizer
except ImportError:
    from models.s4_model import S4ProteinModel, adapt_state_dict_vocab
    from dataloaders.protein import ProteinTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--partial_sequence", type=str, default="MKTAY",
                   help="Partial amino acid sequence to complete")
    p.add_argument("--completion_length", type=int, default=50)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--do_sample", action="store_true", default=True)
    p.add_argument("--greedy", dest="do_sample", action="store_false",
                   help="Use greedy decoding instead of sampling")
    p.add_argument("--stop_at_eos", action="store_true", default=True)
    p.add_argument("--no_stop_at_eos", dest="stop_at_eos", action="store_false")
    p.add_argument("--min_new_tokens", type=int, default=0,
                   help="Do not allow EOS until at least this many residues are generated")
    p.add_argument("--repetition_penalty", type=float, default=1.15,
                   help="Penalty applied to tokens already seen in the generated context")
    p.add_argument("--no_repeat_ngram_size", type=int, default=3,
                   help="Prevent repeating any n-gram of this size")
    return p.parse_args()


def infer_model_shape(state, ckpt):
    model_config = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}
    d_model = state["embed.weight"].shape[1]
    d_state = model_config.get("d_state")
    kernel_type = model_config.get("kernel_type")

    if d_state is None:
        for key in ("blocks.0.s4_layer.kernel.log_A_real", "blocks.0.s4_layer.kernel.A"):
            if key in state:
                d_state = state[key].shape[1] * 2
                break
    if d_state is None:
        d_state = 64

    if kernel_type is None:
        kernel_type = "nplr" if "blocks.0.s4_layer.kernel.A" in state else "diag"

    block_keys = [k for k in state if k.startswith("blocks.")]
    if block_keys:
        indices = [int(k.split(".")[1]) for k in block_keys if len(k.split(".")) >= 2 and k.split(".")[1].isdigit()]
        n_layers = max(indices) + 1 if indices else 6
    else:
        n_layers = 6

    return d_model, d_state, max(1, n_layers), kernel_type


def complete_protein(model, tokenizer, partial_sequence: str, completion_length: int,
                    temperature: float = 1.0, top_k=None, top_p=None, do_sample: bool = True,
                    stop_at_eos: bool = True, repetition_penalty: float = 1.0,
                    no_repeat_ngram_size: Optional[int] = None,
                    min_new_tokens: int = 0, device=None):
    prompt_ids = tokenizer.encode(partial_sequence)
    prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(
        prompt_t,
        max_new_tokens=completion_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
        stop_at_eos=stop_at_eos,
        forbidden_token_ids=(
            tokenizer.pad_token_id,
            tokenizer.mask_token_id,
            tokenizer.unk_token_id,
        ),
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        min_new_tokens=min_new_tokens,
    )
    seq_ids = out[0].tolist()
    return tokenizer.decode(seq_ids, stop_at_eos=stop_at_eos)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    tokenizer = ProteinTokenizer()
    vocab_size = tokenizer.vocab_size
    d_model, d_state, n_layers, kernel_type = infer_model_shape(state, ckpt)

    bidirectional = ckpt.get("bidirectional", False) if isinstance(ckpt, dict) else False
    model = S4ProteinModel(
        vocab_size=vocab_size,
        d_model=d_model,
        d_state=d_state,
        n_layers=n_layers,
        kernel_type=kernel_type,
        bidirectional=bidirectional,
        eos_token_id=tokenizer.eos_token_id,
    )
    state = adapt_state_dict_vocab(state, model.vocab_size)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    completed = complete_protein(
        model,
        tokenizer,
        args.partial_sequence,
        args.completion_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.do_sample,
        stop_at_eos=args.stop_at_eos,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        min_new_tokens=args.min_new_tokens,
        device=device,
    )
    print("Partial:", args.partial_sequence)
    print("Completed:", completed)


if __name__ == "__main__":
    main()

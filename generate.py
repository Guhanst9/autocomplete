"""
Autoregressive protein sequence completion.
"""
import argparse
import os

import torch

try:
    from src.models.s4_model import S4ProteinModel
    from src.dataloaders.protein import ProteinTokenizer
except ImportError:
    from models.s4_model import S4ProteinModel
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
    return p.parse_args()


# autoregressive model that completes a partial protein sequence by generating residues one at a time
def complete_protein(model, tokenizer, partial_sequence: str, completion_length: int,
                    temperature: float = 1.0, top_k=None, top_p=None, do_sample: bool = True,
                    device=None):
    prompt_ids = tokenizer.encode(partial_sequence)
    prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(  # autoregressive generation with sampled decoding
        prompt_t,
        max_new_tokens=completion_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
    )
    seq_ids = out[0].tolist()
    return tokenizer.decode(seq_ids)  # convert token ids back to amino acid sequence string


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    vocab_size = state["embed.weight"].shape[0]
    d_model = state["embed.weight"].shape[1]
    block_keys = [k for k in state if k.startswith("blocks.")]
    if block_keys:
        indices = [int(k.split(".")[1]) for k in block_keys if len(k.split(".")) >= 2 and k.split(".")[1].isdigit()]
        n_layers = max(indices) + 1 if indices else 6
    else:
        n_layers = 6
    n_layers = max(1, n_layers)

    model = S4ProteinModel(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    tokenizer = ProteinTokenizer()
    completed = complete_protein(
        model,
        tokenizer,
        args.partial_sequence,
        args.completion_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.do_sample,
        device=device,
    )
    print("Partial:", args.partial_sequence)
    print("Completed:", completed)


if __name__ == "__main__":
    main()

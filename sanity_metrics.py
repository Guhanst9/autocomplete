"""
Focused sanity checks for protein autocomplete checkpoints.

This is intentionally different from plain full-sequence accuracy. For protein
completion, exact whole-sequence matches are too strict, so this reports
teacher-forced suffix accuracy plus generation quality metrics.
"""
import argparse
from collections import Counter

import torch

try:
    from src.dataloaders.protein import ProteinTokenizer
    from src.models.s4_model import S4ProteinModel, adapt_state_dict_vocab
except ImportError:
    from dataloaders.protein import ProteinTokenizer
    from models.s4_model import S4ProteinModel, adapt_state_dict_vocab


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--target_sequence", required=True,
                   help="Known full protein sequence used for the sanity check")
    p.add_argument("--prefix_length", type=int, default=40)
    p.add_argument("--completion_length", type=int, default=None,
                   help="Defaults to remaining target length plus 20 tokens")
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--repetition_penalty", type=float, default=1.15)
    p.add_argument("--no_repeat_ngram_size", type=int, default=3)
    p.add_argument("--l_max", type=int, default=512)
    return p.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
        indices = [
            int(k.split(".")[1])
            for k in block_keys
            if len(k.split(".")) >= 2 and k.split(".")[1].isdigit()
        ]
        n_layers = max(indices) + 1 if indices else 6
    else:
        n_layers = 6

    return d_model, d_state, max(1, n_layers), kernel_type


def load_model(checkpoint, tokenizer, device):
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    d_model, d_state, n_layers, kernel_type = infer_model_shape(state, ckpt)
    bidirectional = ckpt.get("bidirectional", False) if isinstance(ckpt, dict) else False

    model = S4ProteinModel(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        d_state=d_state,
        n_layers=n_layers,
        kernel_type=kernel_type,
        bidirectional=bidirectional,
        eos_token_id=tokenizer.eos_token_id,
    )
    state = adapt_state_dict_vocab(state, model.vocab_size)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, (d_model, d_state, n_layers, kernel_type, bidirectional)


def composition(sequence):
    counts = Counter(aa for aa in sequence if aa in AMINO_ACIDS)
    total = sum(counts.values())
    if total == 0:
        return {aa: 0.0 for aa in AMINO_ACIDS}
    return {aa: counts[aa] / total for aa in AMINO_ACIDS}


def composition_l1(a, b):
    ca = composition(a)
    cb = composition(b)
    return sum(abs(ca[aa] - cb[aa]) for aa in AMINO_ACIDS)


def longest_homopolymer(sequence):
    best = 0
    current = 0
    prev = None
    for aa in sequence:
        if aa == prev:
            current += 1
        else:
            current = 1
            prev = aa
        best = max(best, current)
    return best


def repeated_ngram_fraction(sequence, n=3):
    if len(sequence) < n:
        return 0.0
    grams = [sequence[i:i + n] for i in range(len(sequence) - n + 1)]
    counts = Counter(grams)
    repeated_extra = sum(count - 1 for count in counts.values() if count > 1)
    return repeated_extra / len(grams)


def valid_aa_fraction(sequence):
    if not sequence:
        return 0.0
    return sum(aa in AMINO_ACIDS for aa in sequence) / len(sequence)


@torch.no_grad()
def teacher_forced_suffix_metrics(model, tokenizer, target_sequence, prefix_length, l_max, device):
    tokens = tokenizer.encode(target_sequence[: l_max - 1]) + [tokenizer.eos_token_id]
    seq_len = len(tokens)
    if prefix_length >= seq_len - 1:
        raise ValueError("prefix_length must leave at least one suffix amino acid")

    input_ids = tokens + [tokenizer.pad_token_id] * (l_max - seq_len)
    target_ids = tokens[1:] + [tokenizer.pad_token_id] + [tokenizer.pad_token_id] * (l_max - seq_len)
    attention_mask = [1] * seq_len + [0] * (l_max - seq_len)

    input_t = torch.tensor([input_ids], dtype=torch.long, device=device)
    target_t = torch.tensor([target_ids], dtype=torch.long, device=device)
    attention_t = torch.tensor([attention_mask], dtype=torch.long, device=device)

    logits = model(input_t, attention_mask=attention_t)
    score_positions = torch.zeros_like(target_t, dtype=torch.bool)
    score_positions[:, prefix_length - 1 : seq_len - 1] = True

    scored_logits = logits[score_positions]
    scored_targets = target_t[score_positions]
    loss = torch.nn.functional.cross_entropy(scored_logits, scored_targets)
    pred = scored_logits.argmax(dim=-1)

    topk = {}
    for k in (1, 3, 5):
        k_eff = min(k, scored_logits.shape[-1])
        top = scored_logits.topk(k_eff, dim=-1).indices
        topk[k] = (top == scored_targets.unsqueeze(-1)).any(dim=-1).float().mean().item()

    eos_target_mask = scored_targets == tokenizer.eos_token_id
    eos_accuracy = None
    if eos_target_mask.any().item():
        eos_accuracy = (pred[eos_target_mask] == tokenizer.eos_token_id).float().mean().item()

    return {
        "suffix_loss": loss.item(),
        "suffix_perplexity": torch.exp(loss).item(),
        "suffix_top1": topk[1],
        "suffix_top3": topk[3],
        "suffix_top5": topk[5],
        "scored_suffix_tokens": int(score_positions.sum().item()),
        "eos_accuracy": eos_accuracy,
    }


@torch.no_grad()
def generate_one(model, tokenizer, prefix, completion_length, args, device):
    prompt = torch.tensor([tokenizer.encode(prefix)], dtype=torch.long, device=device)
    out = model.generate(
        prompt,
        max_new_tokens=completion_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=not args.greedy,
        eos_token_id=tokenizer.eos_token_id,
        stop_at_eos=True,
        forbidden_token_ids=(
            tokenizer.pad_token_id,
            tokenizer.mask_token_id,
            tokenizer.unk_token_id,
        ),
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    ids = out[0].tolist()
    new_ids = ids[len(prompt[0]):]
    completed = tokenizer.decode(ids, stop_at_eos=True)
    eos_generated = tokenizer.eos_token_id in new_ids
    return completed, eos_generated


def generation_metrics(completed, eos_generated, prefix, target_suffix):
    generated_suffix = completed[len(prefix):] if completed.startswith(prefix) else completed
    overlap = min(len(generated_suffix), len(target_suffix))
    positional_identity = 0.0
    if overlap:
        positional_identity = sum(
            a == b for a, b in zip(generated_suffix[:overlap], target_suffix[:overlap])
        ) / overlap

    adjacent_repeat_fraction = 0.0
    if len(generated_suffix) > 1:
        adjacent_repeat_fraction = sum(
            generated_suffix[i] == generated_suffix[i - 1]
            for i in range(1, len(generated_suffix))
        ) / (len(generated_suffix) - 1)

    return {
        "generated_suffix": generated_suffix,
        "generated_suffix_len": len(generated_suffix),
        "target_suffix_len": len(target_suffix),
        "length_error": len(generated_suffix) - len(target_suffix),
        "eos_generated": eos_generated,
        "positional_identity_overlap": positional_identity,
        "composition_l1_vs_target": composition_l1(generated_suffix, target_suffix),
        "valid_aa_fraction": valid_aa_fraction(generated_suffix),
        "longest_homopolymer": longest_homopolymer(generated_suffix),
        "adjacent_repeat_fraction": adjacent_repeat_fraction,
        "repeated_3gram_fraction": repeated_ngram_fraction(generated_suffix, 3),
    }


def print_mean(name, values, fmt=".4f"):
    if not values:
        return
    mean = sum(values) / len(values)
    print(f"{name}: {mean:{fmt}}")


def main():
    args = parse_args()
    tokenizer = ProteinTokenizer()
    device = get_device()
    model, shape = load_model(args.checkpoint, tokenizer, device)

    target = args.target_sequence.strip().upper()
    prefix = target[:args.prefix_length]
    target_suffix = target[args.prefix_length:]
    completion_length = args.completion_length
    if completion_length is None:
        completion_length = len(target_suffix) + 20

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(
        "Model: "
        f"d_model={shape[0]}, d_state={shape[1]}, n_layers={shape[2]}, "
        f"kernel_type={shape[3]}, bidirectional={shape[4]}"
    )
    print(f"Prefix length: {len(prefix)}")
    print(f"Target length: {len(target)}")
    print(f"Completion budget: {completion_length}")
    print(f"Prefix: {prefix}")
    print()

    tf = teacher_forced_suffix_metrics(
        model, tokenizer, target, len(prefix), args.l_max, device
    )
    print("Teacher-forced suffix metrics")
    print(f"  suffix loss: {tf['suffix_loss']:.4f}")
    print(f"  suffix perplexity: {tf['suffix_perplexity']:.4f}")
    print(f"  suffix top-1 accuracy: {tf['suffix_top1']:.4f}")
    print(f"  suffix top-3 accuracy: {tf['suffix_top3']:.4f}")
    print(f"  suffix top-5 accuracy: {tf['suffix_top5']:.4f}")
    print(f"  scored suffix tokens: {tf['scored_suffix_tokens']}")
    if tf["eos_accuracy"] is not None:
        print(f"  EOS accuracy: {tf['eos_accuracy']:.4f}")
    print()

    sample_metrics = []
    print("Generated samples")
    for i in range(args.num_samples):
        completed, eos_generated = generate_one(
            model, tokenizer, prefix, completion_length, args, device
        )
        metrics = generation_metrics(completed, eos_generated, prefix, target_suffix)
        sample_metrics.append(metrics)
        print(f"  sample {i + 1}: {completed}")
        print(
            "    "
            f"suffix_len={metrics['generated_suffix_len']}, "
            f"eos={metrics['eos_generated']}, "
            f"identity={metrics['positional_identity_overlap']:.4f}, "
            f"composition_l1={metrics['composition_l1_vs_target']:.4f}, "
            f"longest_repeat={metrics['longest_homopolymer']}, "
            f"repeat_3gram={metrics['repeated_3gram_fraction']:.4f}"
        )

    print()
    print("Generation summary")
    print_mean("  mean positional identity over overlap", [m["positional_identity_overlap"] for m in sample_metrics])
    print_mean("  mean composition L1 vs target", [m["composition_l1_vs_target"] for m in sample_metrics])
    print_mean("  mean generated suffix length", [m["generated_suffix_len"] for m in sample_metrics], ".2f")
    print_mean("  mean length error", [m["length_error"] for m in sample_metrics], ".2f")
    print_mean("  mean adjacent repeat fraction", [m["adjacent_repeat_fraction"] for m in sample_metrics])
    print_mean("  mean repeated 3-gram fraction", [m["repeated_3gram_fraction"] for m in sample_metrics])
    eos_rate = sum(m["eos_generated"] for m in sample_metrics) / len(sample_metrics)
    print(f"  EOS generation rate: {eos_rate:.4f}")


if __name__ == "__main__":
    main()

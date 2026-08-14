import argparse
from pathlib import Path

import torch

from src.dna.checkpoint import load_model


DEFAULT_CHECKPOINT = "outputs/plastid_s4d_v2_recovery_full/best_loss.pt"


def clean_prompt(sequence: str) -> str:
    prompt = "".join(sequence.split()).upper()
    if not prompt:
        raise ValueError("prompt cannot be empty")
    invalid = sorted(set(prompt) - set("ACGT"))
    if invalid:
        raise ValueError(f"prompt contains invalid DNA bases: {', '.join(invalid)}")
    return prompt


def read_prompt_file(path: str) -> str:
    lines = Path(path).read_text().splitlines()
    sequence_lines = []
    saw_header = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if saw_header:
                break
            saw_header = True
            continue
        sequence_lines.append(line)

    return clean_prompt("".join(sequence_lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a DNA continuation from a trained DNA checkpoint."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="DNA context containing only A, C, G, and T")
    prompt_group.add_argument("--prompt-file", help="text or FASTA file containing the prompt")
    parser.add_argument("--max-new-bases", type=int, required=True)
    parser.add_argument(
        "--decoding-mode",
        choices=("greedy", "sampled"),
        default="sampled",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_new_bases <= 0:
        raise ValueError("max-new-bases must be positive")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")

    prompt = (
        clean_prompt(args.prompt)
        if args.prompt is not None
        else read_prompt_file(args.prompt_file)
    )
    model, tokenizer, device = load_model(args.checkpoint)
    prompt_ids = tokenizer.encode(prompt)
    if tokenizer.unk_token_id in prompt_ids:
        raise ValueError("checkpoint tokenizer could not encode the prompt")

    torch.manual_seed(args.seed)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    forbidden = {
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
        tokenizer.eos_token_id,
    }
    n_token_id = tokenizer.vocab.get("N")
    if n_token_id is not None:
        forbidden.add(n_token_id)

    output = model.generate(
        prompt_tensor,
        max_new_tokens=args.max_new_bases,
        use_recurrent=True,
        stop_at_eos=False,
        forbidden_token_ids=tuple(sorted(forbidden)),
        min_new_tokens=args.max_new_bases,
        sampling_temperature=args.temperature if args.decoding_mode == "sampled" else None,
    )
    generated_ids = output[0, len(prompt_ids) :].tolist()
    generated = tokenizer.decode(generated_ids, stop_at_eos=False)

    print("DNA generation")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device: {device}")
    print(f"  Model type: {getattr(model, 'model_type', 's4d')}")
    print(f"  Prompt length: {len(prompt)}")
    print(f"  Generated length: {len(generated)}")
    print(f"  Decoding mode: {args.decoding_mode}")
    if args.decoding_mode == "sampled":
        print(f"  Temperature: {args.temperature}")
        print(f"  Seed: {args.seed}")
    print()
    print(f'prompt = "{prompt}"')
    print(f'generated_suffix = "{generated}"')
    print(f'full_sequence = "{prompt}{generated}"')


if __name__ == "__main__":
    main()

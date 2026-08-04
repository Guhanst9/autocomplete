# S4D DNA sequence prediction

This project trains an S4D model to predict the next DNA base in plastid genomes. The active code supports DNA training, direct continuation from a prompt, sliding-window evaluation, and synthetic controls. The earlier protein experiments are preserved under `legacy/protein/`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Model checkpoints and FASTA files are kept outside Git because they are large. The main checkpoint used for the current experiments is:

```text
outputs/plastid_s4d_v2_recovery_full/best_loss.pt
```

## Generate DNA

`generate_dna.py` accepts any nonempty A/C/G/T prompt and any positive generation limit.

```bash
python generate_dna.py \
  --checkpoint outputs/plastid_s4d_v2_recovery_full/best_loss.pt \
  --prompt ACGTTGCAACGTTGCA \
  --max-new-bases 256 \
  --decoding-mode sampled \
  --temperature 0.8 \
  --seed 13
```

For a longer prompt, put the sequence in a text or FASTA file:

```bash
python generate_dna.py \
  --checkpoint outputs/plastid_s4d_v2_recovery_full/best_loss.pt \
  --prompt-file data/prompt.fna \
  --max-new-bases 512
```

Use `--decoding-mode greedy` to always choose the most likely next base. Sampled decoding at temperature `0.8` produced fewer repeat collapses in the Rosa evaluation.

## Sliding evaluation

```bash
python run_sliding_eval.py \
  --checkpoint outputs/plastid_s4d_v2_recovery_full/best_loss.pt \
  --fasta_file data/plastid/refseq_full/refseq_plastids_all.fna.gz \
  --accession NC_053550.1 \
  --prompt_length 512 \
  --generate_length 512 \
  --stride 256 \
  --circular \
  --batch_size 4 \
  --decoding_mode sampled \
  --temperature 0.8 \
  --seed 13 \
  --output_dir outputs/rosa_eval
```

## Train

```bash
python run_plastid.py \
  --preset full \
  --fasta-file data/plastid/refseq_full/refseq_plastids_all_clean_no_n.fna.gz \
  --output-dir outputs/plastid_s4d_v2_recovery_full \
  --holdout-accession NC_053550.1
```

Available presets are `quick-control`, `quick`, and `full`.

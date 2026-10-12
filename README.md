# S4D plastid DNA

This repository contains code for training and evaluating sequence models on plant plastid DNA. The main experiments compare three S4D model sizes with a parameter-matched Transformer. Models can predict one base or one triplet at a time. Older protein experiments are in `legacy/protein/`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

FASTA files and model checkpoints are not stored in Git. By default, training looks for:

```text
data/plastid/refseq_full/refseq_plastids_all_clean_no_n.fna.gz
```

Pass `--fasta-file` to use a different dataset.

## Train

Example triplet-prediction run:

```bash
python run_plastid.py \
  --preset full-recovery \
  --prediction-unit triplet \
  --output-dir outputs/my-triplet-run
```

The best validation-loss checkpoint is saved as `best_loss.pt` in the output directory.

## Generate a continuation

```bash
python generate_dna.py \
  --checkpoint outputs/my-triplet-run/best_loss.pt \
  --prompt ACGTTGCAACGTTGCA \
  --max-new-bases 120 \
  --decoding-mode sampled \
  --temperature 0.8 \
  --seed 13
```

Use `--decoding-mode greedy` to choose the most probable output each step.

## Evaluation

`prepare_test_panel.py` prepares a test panel. `evaluate_models.py` runs the model comparison, and `run_sliding_eval.py` evaluates a checkpoint on genome windows. Run each script with `--help` for options.

One-off analyses, benchmarks, and figure scripts are in `scripts/analysis/`. Run them from the repository root as modules, for example `python -m scripts.analysis.make_final_figures --help`.

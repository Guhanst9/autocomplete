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

Current checkpoint SHA-256:

```text
67b2c1f3b73261ea7f30c0e6f746b547286b1715a7c2a1498f6ed54f18db639b
```

Checkpoint download link:

```text
TODO: add Google Drive or Dropbox link for best_loss.pt
```

Do not commit model checkpoints or full FASTA files. The full training FASTA is hundreds of MB and should stay local.

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

Greedy decoding is useful as a stability test because it always takes the highest-probability next base. Sampled decoding at temperature `0.8` adds controlled randomness and removed the long same-base repeat collapse in the Rosa sampled run, while lowering exact accuracy slightly.

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

Important output columns:

```text
prompt             input DNA given to the model
generated_suffix   model-generated continuation
true_suffix        real reference continuation
accuracy_percent   exact base-by-base identity between generated_suffix and true_suffix
region             inferred chloroplast region label
longest_generated_run  longest same-base run in generated_suffix
n_count            number of N bases generated
gc_difference_percent  absolute GC-content difference from the true suffix
```

## Biological evaluation

The biological evaluation freezes the current checkpoint and studies its outputs without retraining.

Prepare baseline metadata and the panel manifest:

```bash
python run_biological_eval.py \
  --config configs/plastid_evaluation.yaml \
  --stage prepare \
  --output-dir outputs/plastid_biological_eval
```

Download/cache GenBank annotations for the first panel genome:

```bash
python run_biological_eval.py \
  --config configs/plastid_evaluation.yaml \
  --stage annotations \
  --output-dir outputs/plastid_biological_eval \
  --max-genomes 1
```

Run a small smoke evaluation:

```bash
python run_biological_eval.py \
  --config configs/plastid_evaluation.yaml \
  --stage sliding \
  --output-dir outputs/plastid_biological_eval \
  --max-genomes 1 \
  --max-windows 2 \
  --seeds 13
```

Create compact summaries, figures, and the Markdown report:

```bash
python run_biological_eval.py \
  --config configs/plastid_evaluation.yaml \
  --stage summarize \
  --output-dir outputs/plastid_biological_eval \
  --reports-dir reports/plastid_eval

python run_biological_eval.py \
  --config configs/plastid_evaluation.yaml \
  --stage figures \
  --reports-dir reports/plastid_eval

python run_biological_eval.py \
  --config configs/plastid_evaluation.yaml \
  --stage report \
  --reports-dir reports/plastid_eval
```

The full generated sequence CSVs stay in ignored `outputs/`. Compact tables and figures are tracked under `reports/`.

Model limitations:

```text
The model predicts DNA bases from local sequence context. It is not explicitly gene-aware, does not know reading frames during generation, and was trained on 1,024-base windows. Long-range plastome organization should be treated as exploratory.
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

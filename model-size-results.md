# S4D model-size comparison

The model-size experiment tests whether adding parameters improves triplet prediction and recursive DNA generation. All three models use 10 S4D layers and a state size of 64. Only the model width changes, which keeps the comparison focused on model capacity rather than depth or a different training procedure.

## Shared training setup

All three runs use the same settings:

```text
prediction unit:       3 bases (64 possible triplets)
S4D layers:            10
state size:            64
context length:        1,024 bases
training windows:      54,000 per epoch
validation windows:    6,000 fixed windows
windows per genome:    4 newly sampled training windows per epoch
batch size:            8
learning rate:         0.0003
dropout:               0.1
maximum epochs:        20
early stopping:        patience 3, minimum improvement 0.002
random seed:           13
decoding:              sampled, temperature 0.8, seed 13
held-out genome:       NC_053550.1 Rosa minutifolia
```

Training uses the cleaned no-`N` plastid FASTA. Training records are separated from validation records, validation windows remain fixed, and `NC_053550.1` is excluded from both. The training locations are resampled each epoch so the model does not repeatedly see only the same four sections of each genome.

The model configurations are:

| Preset | Width | Layers | State size | Parameters |
|---|---:|---:|---:|---:|
| `size-small` | 196 | 10 | 64 | 4,134,228 |
| `size-current` | 400 | 10 | 64 | 16,597,200 |
| `size-large` | 512 | 10 | 64 | 26,978,816 |

## Comparison metrics

- **Validation loss** is triplet cross-entropy on held-out windows. Lower is better.
- **Base perplexity** converts triplet loss into a per-base value. A lower value means the model is less uncertain about the next base.
- **Per-base accuracy** measures how many individual bases are correct inside the predicted triplets when the true previous sequence is available.
- **Exact triplet accuracy** requires all three bases in a predicted triplet to be correct.
- **Rosa accuracy** is the harder recursive test. The model receives 512 real bases, generates the next 512 bases, and is compared position by position with the true continuation across 614 circular windows.
- **First 100** and **final 100** show how accuracy changes as generated predictions become part of the model's next input.

## Results

The primary comparison uses `best_loss.pt` from each run and the same sampled Rosa evaluation.

| Model | Validation loss | Base perplexity | Per-base accuracy | Exact triplet accuracy | Rosa accuracy | First 100 | Final 100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4.13M | 1.5289 | 1.665 | 78.67% | 62.21% | 58.8536% | 73.9202% | 49.1498% |
| 16.60M | 1.1809 | 1.482 | 84.05% | 70.40% | 61.3666% | 79.0603% | 49.8713% |
| 26.98M | **1.1011** | **1.443** | **85.20%** | **72.30%** | **62.2958%** | **80.9121%** | **50.0212%** |

All three models completed 20 epochs because validation loss continued improving and never met the early-stopping condition. The 26.98M model had the strongest validation and Rosa results, but the benefit became smaller as the models grew:

```text
4.13M to 16.60M:  validation loss -0.3480, Rosa accuracy +2.5130 points
16.60M to 26.98M: validation loss -0.0798, Rosa accuracy +0.9292 points
```

The larger models improved the beginning of each generated continuation much more than the end. Final-100 accuracy increased from 49.15% to only 50.02% across the entire size range. This suggests that increasing model size alone does not solve the accumulation of errors during recursive generation.

## Runtime and cost

The AWS runs used one `g5.xlarge` instance with a single NVIDIA A10G GPU. The measured on-demand compute price was approximately `$1.006` per hour.

| Model | Training time | Rosa evaluation | Approximate successful-run cost |
|---|---:|---:|---:|
| 4.13M | 10h 05m across two sessions | 7m 22s | $10.37 |
| 16.60M | 16h 22m | 7m 52s | about $17.00 |
| 26.98M | 20h 01m | 7m 47s | $20.26 |

The 26.98M experiment also had an interrupted first attempt. Including that attempt, its total instance usage was about 21h 25m and `$21.55`. The table uses the successful run where possible so startup failures do not distort the model comparison.

The 4.13M model was the least expensive, but the 16.60M model provided the largest improvement per size increase. The 26.98M model was best overall, although it cost more and produced a smaller additional accuracy gain.

## Accessibility

The active training code supports both Apple MPS and NVIDIA CUDA. The models can therefore run locally on an Apple Silicon computer, although a single A10G cloud GPU trains them much faster. Earlier benchmarks measured the A10G at roughly 3.5 times the local Mac training speed for a comparable S4D model.

This project is intentionally smaller than large genome foundation models. It uses one computer or one rented GPU rather than a large GPU cluster. The 4.13M model is useful for lower-cost experiments, while the 16.60M model gives a stronger result without requiring the largest tested configuration. This makes it possible for students and independent researchers to reproduce the pipeline, change the dataset, or test another biological sequence without Stanford- or NVIDIA-scale computing resources.

## Reproducing the runs

Replace `PRESET` and `OUTPUT_DIR` with the desired model size:

```bash
python run_plastid.py \
  --preset PRESET \
  --prediction-unit triplet \
  --fasta-file data/plastid/refseq_full/refseq_plastids_all_clean_no_n.fna.gz \
  --output-dir OUTPUT_DIR \
  --max-additional-epochs 20 \
  --early-stopping-patience 3 \
  --early-stopping-min-delta 0.002 \
  --batch-size 8 \
  --holdout-accession NC_053550.1
```

Use one of these preset/output pairs:

```text
size-small    outputs/aws_s4d_triplet_size_small
size-current  outputs/s4d_triplet_16.60m_resampled_4windows
size-large    outputs/aws_s4d_triplet_size_large
```

Checkpoints and full generated CSV files are ignored by Git and must be stored separately.

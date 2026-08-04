# Plastid S4D evaluation

## Motivation and question

This report freezes the current S4D plastid checkpoint and studies what it generates before changing training again. The main question is whether the model's generated DNA preserves local sequence identity, region behavior, annotated genes/RNAs, inverted repeats, and context-length effects.

## Dataset and cleaning

The model uses cleaned plastid FASTA data with high-N records removed and remaining N bases replaced before training. The frozen checkpoint is `outputs/plastid_s4d_v2_recovery_full/best_loss.pt`. `NC_053550.1` is treated as a decoding-development genome because it was used while choosing sampled decoding at temperature `0.8`.

## Model setup

The checkpoint is the 16.57M-parameter S4D-v2 model trained with 1,024-base context, next-base prediction, two-pass recovery, and a weak homopolymer-ending loss. The primary generation mode for evaluation is sampled decoding at temperature `0.8`; greedy decoding is kept as a stability diagnostic.

## Decoding comparison

| decoding_mode | seed | rows | avg_accuracy_percent | min_accuracy_percent | max_accuracy_percent | avg_gc_difference_percent | max_longest_generated_run | runs_over_20 | n_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sampled | 13 | 2 | 96.00 | 95.90 | 96.09 | 0.88 | 5 | 0 | 0 |

## Genome-region results

| accession | group | exposure | decoding_mode | seed | region | rows | avg_accuracy_percent | min_accuracy_percent | max_accuracy_percent | avg_gc_difference_percent | max_longest_generated_run | runs_over_20 | n_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NC_053550.1 | Rosa | decoding-development | sampled | 13 | LSC | 2 | 96.00 | 95.90 | 96.09 | 0.88 | 5 | 0 | 0 |

## CDS, tRNA, and rRNA results

| accession | feature_label | feature_type | rows | avg_accuracy_percent | avg_aa_identity_percent | internal_stop_codons | total_overlap_bases |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NC_053550.1 | psbA | CDS | 2 | 96.00 | 90.00 | 15 | 1024 |

## Inverted-repeat results

| accession | rows | avg_true_ir_identity_percent | avg_generated_ir_identity_percent | avg_ira_identity_to_true_percent | avg_irb_identity_to_true_percent |
| --- | --- | --- | --- | --- | --- |
| NC_053550.1 | 2 | 24.02 | 26.56 | 41.02 | 66.60 |

## Context-length and top-k results

| accession | target_start | bases | top1_accuracy_percent | top2_accuracy_percent | mean_true_base_probability | base_frequency_baseline_percent | baseline_base |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NC_053550.1 | 512 | 512 | 96.68 | 99.80 | 0.9517 | 25.20 | T |
| NC_053550.1 | 768 | 512 | 96.48 | 100.00 | 0.9514 | 22.27 | T |

Context-length outputs are stored in ignored CSV files under `outputs/plastid_biological_eval/` because they include generated sequences.

## Synthetic controls

Synthetic evaluation remains a separate check through `run_synthetic_eval.py`. It is useful for debugging repeated-base behavior without changing the biological evaluation panel.

## Limitations and next experiments

The current report is based on local panel smoke outputs unless the full stages are rerun. GenBank annotations are used for feature labels, while IRA/IRB currently falls back to sequence inference when annotation boundaries are unavailable. External NCBI test genomes and model-size comparisons should happen only after the local panel evaluation is reviewed.

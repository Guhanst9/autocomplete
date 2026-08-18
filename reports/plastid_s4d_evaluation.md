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
| sampled | 13 | 4101 | 50.69 | 19.53 | 100.00 | 10.09 | 17 | 0 | 0 |

## Genome-region results

| accession | group | exposure | decoding_mode | seed | region | rows | avg_accuracy_percent | min_accuracy_percent | max_accuracy_percent | avg_gc_difference_percent | max_longest_generated_run | runs_over_20 | n_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NC_023110.1 | Sunflower | validation | sampled | 13 | IRA | 95 | 68.54 | 23.83 | 99.80 | 3.58 | 15 | 0 | 0 |
| NC_023110.1 | Sunflower | validation | sampled | 13 | IRB | 94 | 66.10 | 24.02 | 99.80 | 2.94 | 10 | 0 | 0 |
| NC_023110.1 | Sunflower | validation | sampled | 13 | LSC | 324 | 46.61 | 23.63 | 97.85 | 13.04 | 17 | 0 | 0 |
| NC_023110.1 | Sunflower | validation | sampled | 13 | SSC | 69 | 32.58 | 27.34 | 42.19 | 14.60 | 12 | 0 | 0 |
| NC_023110.1 | Sunflower | validation | sampled | 13 | boundary | 8 | 52.12 | 33.01 | 86.72 | 8.33 | 14 | 0 | 0 |

## CDS, tRNA, and rRNA results

| accession | feature_label | feature_type | rows | avg_accuracy_percent | avg_aa_identity_percent | internal_stop_codons | total_overlap_bases |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NC_023110.1 | all_CDS | CDS | 461 | 52.45 | 30.83 | 1361 | 140154 |
| NC_023110.1 | all_rRNA | rRNA | 12 | 51.95 |  |  | 1094 |
| NC_023110.1 | all_tRNA | tRNA | 95 | 38.95 |  |  | 5276 |
| NC_023110.1 | matK | CDS | 7 | 44.56 | 18.90 | 37 | 3006 |
| NC_023110.1 | psbA | CDS | 6 | 70.73 | 100.00 | 0 | 1860 |

## Inverted-repeat results

| accession | rows | avg_true_ir_identity_percent | avg_generated_ir_identity_percent | avg_ira_identity_to_true_percent | avg_irb_identity_to_true_percent |
| --- | --- | --- | --- | --- | --- |
| NC_023110.1 | 20 | 100.00 | 56.13 | 69.39 | 67.09 |
| NC_027476.1 | 20 | 100.00 | 34.77 | 43.68 | 48.54 |
| NC_030275.1 | 20 | 100.00 | 48.60 | 70.61 | 65.38 |
| NC_030377.1 | 20 | 100.00 | 37.78 | 49.65 | 46.96 |
| NC_038102.1 | 20 | 100.00 | 51.84 | 65.31 | 67.33 |

## Context-length and top-k results

Free-generation context-length comparison:

| accession | context_length | rows | avg_accuracy_percent | min_accuracy_percent | max_accuracy_percent | avg_gc_difference_percent | max_longest_run | avg_kmer_diversity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NC_023110.1 | 128 | 5 | 72.50 | 33.40 | 97.07 | 4.37 | 9 | 0.9830 |
| NC_023110.1 | 256 | 5 | 78.24 | 32.23 | 97.07 | 2.38 | 6 | 0.9802 |
| NC_023110.1 | 512 | 5 | 78.67 | 33.79 | 97.27 | 1.17 | 9 | 0.9826 |
| NC_023110.1 | 1024 | 5 | 79.42 | 35.16 | 97.07 | 4.38 | 8 | 0.9533 |
| NC_027476.1 | 128 | 5 | 52.66 | 28.52 | 94.34 | 7.34 | 8 | 0.9715 |

Teacher-forced top-k comparison:

| accession | target_start | bases | top1_accuracy_percent | top2_accuracy_percent | mean_true_base_probability | base_frequency_baseline_percent | baseline_base |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NC_053550.1 | 512 | 512 | 96.68 | 99.80 | 0.9517 | 25.20 | T |
| NC_053550.1 | 768 | 512 | 96.48 | 100.00 | 0.9514 | 22.27 | T |
| NC_053550.1 | 1024 | 512 | 95.70 | 99.80 | 0.9449 | 21.88 | T |
| NC_053550.1 | 1280 | 512 | 84.77 | 94.34 | 0.7982 | 29.88 | T |
| NC_053550.1 | 1536 | 512 | 79.30 | 91.80 | 0.7134 | 32.42 | T |

Full context-length outputs are stored in ignored CSV files under `outputs/plastid_biological_eval/` because they include generated sequences.

## Synthetic controls

Synthetic evaluation remains a separate check through `run_synthetic_eval.py`. It is useful for debugging repeated-base behavior without changing the biological evaluation panel.

## Limitations and next experiments

The current report is based on the full local panel evaluation for the seven selected plastid genomes. GenBank repeat boundaries are used when available. Other genomes use mismatch-tolerant sequence inference, and each generated CSV records the boundary source. External NCBI test genomes and model-size comparisons should happen only after the local panel evaluation is reviewed.

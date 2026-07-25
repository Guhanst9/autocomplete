# legacy protein workflow

This folder keeps the original protein autocomplete and masked-language-model workflow runnable after the active repo moved to plastid DNA prediction.

Run commands from this folder:

```bash
cd legacy/protein
python train.py --objective autocomplete --fasta_file ../../data/protein/uniref50.fasta.gz
python generate.py --checkpoint ../../outputs/path/to/best.pt --partial_sequence MKTAY
python test_protein_objective.py
python test_s4_kernel.py
```

The archive keeps the old NPLR/HiPPO helper code because the protein experiments and tests used it. Active DNA training no longer depends on those files.

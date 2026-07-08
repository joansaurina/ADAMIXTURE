# Demo

This demo runs ADAMIXTURE on a small PLINK BED dataset with 105 individuals and
8451 variants. It is intended as a quick installation/debugging check, not as a
scientific analysis.

From this directory, run:

```console
sh run_demo.sh
```

The script runs the installed `adamixture` command on CPU with the BRQN
optimizer and compares the generated `P` and `Q` files against expected demo
outputs.

## Reader fixtures

The `data/` directory contains the same small dataset in BED, PGEN and VCF
forms, plus compressed variants used by the reader tests:

- `.bed`, `.bed.gz`, `.bed.zst` with `.bim`/`.fam` sidecars in plain, `.gz`
  and `.zst` forms.
- `.pgen` and `.pgen.zst` with `.pvar`/`.psam` sidecars in plain, `.gz` and
  `.zst` forms.
- `.vcf`, `.vcf.gz` and `.vcf.zst`.

## Regenerating expected outputs

CPU reader/optimizer fixtures are regenerated through the pytest update mode:

```console
ADAMIXTURE_UPDATE_EXPECTED=1 python -m pytest tests
```

Device-specific fixtures can be generated from this checkout:

```console
python adamixture/demo/generate_device_expected.py --device cuda
python adamixture/demo/generate_device_expected.py --device mps
```

The CUDA/MPS pytest files are optional and only run when explicitly enabled:

```console
ADAMIXTURE_TEST_CUDA=1 python -m pytest tests/test_cuda.py
ADAMIXTURE_TEST_MPS=1 python -m pytest tests/test_mps.py
```

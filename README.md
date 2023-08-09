# shiprec

**s**calable **h**istopathology **i**mage **p**rerocessing and featu**re** extra**c**tion

## Installation
```bash
apt install openslide-tools
python -m venv env
source env/bin/activate
pip install numpy wheel Cython
source env/bin/activate
```

## Configuration
See `config.yaml`.

## Running
```bash
env/bin/python -m shiprec.run
```
You may provide command-line overrides of specific configuration options from `config.yaml` (see [this documentation](https://hydra.cc/docs/advanced/override_grammar/basic/)).

## Benchmark

I ran a benchmark with the following setup on the first 10 slides of TCGA-BRCA:
- 16 workers
- 1 GPU
- 2 slides in parallel

### Results
- **total time for 10 slides:** 1862 seconds (~31.0 minutes)
- **average time per slide (parallel):** 186 seconds (~3.1 minutes)
- **actual average time per slide:** 351 seconds (~5.9 minutes) [this is about twice the parallel time because we compute 2 slides in parallel]


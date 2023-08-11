# shiprec

**s**calable **h**istopathology **i**mage **p**reprocessing and featu**re** extra**c**tion

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

I ran a benchmark on the first 10 slides of TCGA-BRCA with various setups, as well as the [E2E pipeline](https://github.com/KatherLab/end2end-WSI-preprocessing).
All experiments were run on the same planet computer.


| **SETUP**                                            | E2E      | A        | B        | C        | D        | E        |
| ---------------------------------------------------- | -------- | -------- | -------- | -------- | -------- | -------- |
| number of CPU workers                                | 8        | 8        | 8        | 16       | 16       | 16       |
| number of slides in parallel                         | 1        | 1        | 2        | 2        | 4        | 8        |
| GPUs                                                 | 1        | 1        | 1        | 1        | 1        | 1        |
| save image files                                     | no*      | yes      | yes      | yes      | yes      | yes      |
| **RESULTS**                                          |
| total time for 10 slides (minutes)                   | 54.3     | 22.5     | 21.0     | 18.8     | 16.8     | 15.8     |
| average time per slide (= total time / 10) (minutes) | 5.43     | 2.25     | 2.10     | 1.88     | 1.68     | 1.58     |
| average time per slide start to finish (minutes)     | 5.43     | 2.25     | 4.02     | 3.67     | 5.57     | 9.72     |
| **est. throughput (slides/hr)**                      | **11.1** | **26.7** | **28.6** | **31.9** | **35.7** | **38.0** |


(* removed image saving feature from E2E pipeline to make it faster)
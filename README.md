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
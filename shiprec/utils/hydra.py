import hydra
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval)  # ensure that eval is available in the config file

__all__ = ["hydra"]

from omegaconf import OmegaConf, DictConfig
from pathlib import Path

from .utils import hydra


@hydra.main(config_path=str(Path(__file__).parent.parent), config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    main()

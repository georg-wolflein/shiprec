from loguru import logger
from tqdm import tqdm

# Make tqdm work with loguru
logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

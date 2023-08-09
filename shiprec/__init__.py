from loguru import logger
from tqdm import tqdm
import logging


def formatter(record):
    if "slide_id" in record.get("extra", []):
        return (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            # "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<cyan>slide {extra[slide_id]}</cyan> - <level>{message}</level>\n"
        )
    else:
        return (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            # "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>\n"
        )


# Make tqdm work with loguru
logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, format=formatter)

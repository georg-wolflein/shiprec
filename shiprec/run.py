import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import dask.array as da
import dask
import torch
from dask.distributed import Client
from loguru import logger
from concurrent.futures import Future
from torch import nn
from typing import NamedTuple, Sequence, Union
from time import time

import shiprec
from shiprec.slide import load_slide, Backend as SlideBackend
from shiprec.progress import tqdm_dask
from shiprec.cluster import MixedCluster
from shiprec.patching import split_slide_into_patches, flatten_patches_and_coords
from shiprec.canny import get_canny_foreground_mask
from shiprec.macenko.dask import DaskMacenkoNormalizer
from shiprec.feature_extraction import load_model, extract_features
from shiprec.utils import fix_dask_torch  # noqa

SLIDE_FILE = "/data/data/TCGA-BRCA-DX-IMGS_1133/TCGA-AO-A12B-01Z-00-DX1.B215230B-5FF7-4B0A-9C1E-5F1658534B11.svs"
TARGET_PATCH_FILE = "/app/normalization_template.jpg"


class PipelineSetup(NamedTuple):
    client: Client
    stain_target: da.Array
    model: Future[nn.Module]
    has_gpu_workers: bool


def setup_pipeline(n_cpu_workers: int = 8, gpu_devices: Sequence[int] = (), memory_limit="32GB") -> PipelineSetup:
    # Start the Dask cluster
    cluster = MixedCluster(n_cpu_workers=n_cpu_workers, gpu_devices=gpu_devices, memory_limit=memory_limit)
    client = Client(cluster.scheduler_address)
    gpu_workers = [
        (w.host, w.port) for w in cluster.workers.values() if w.resources is not None and w.resources.get("GPU", 0) > 0
    ]
    logger.info(f"Started Dask client with {len(cluster.workers)} workers, {len(gpu_workers)} of which are GPU workers")
    logger.info(f"Access the Dask dashboard at {cluster.dashboard_link}")

    # Load the stain target
    stain_target = cv2.cvtColor(cv2.imread(str(TARGET_PATCH_FILE)), cv2.COLOR_BGR2RGB)
    stain_target = da.from_array(stain_target)
    stain_target = client.persist(stain_target)

    # Load feature extraction model
    logger.info("Loading feature extraction model")
    model = load_model()
    model.eval()

    # Send the model to the workers
    # TODO: send the model to the GPU workers only (currently it is sent to all workers)
    dmodel = client.scatter(model)
    logger.info("Loaded and scattered feature extractor to workers")
    return PipelineSetup(client, stain_target, dmodel, has_gpu_workers=len(gpu_devices) > 0)


def process_slide(
    pipeline_setup: PipelineSetup,
    slide_file: Path,
    patch_size: int = 224,
    target_mpp=256.0 / 224.0,
    macenko_batch_size: int = 256,
    feature_extraction_batch_size: int = 256,  # recommended to be a multiple or divisor of macenko_batch_size
    synchronous_saving: bool = False,
    slide_backend: SlideBackend = "cucim",
) -> Union[Future, Sequence[Future], None]:
    """Process a slide and save the features to disk."""

    client, stain_target, model, has_gpu_devices = pipeline_setup

    # Load the slide
    slide = load_slide(slide_file, target_mpp=target_mpp, backend=slide_backend)

    # Split the slide into patches
    patches, patch_coords = split_slide_into_patches(slide, patch_size=patch_size)
    patches, patch_coords = flatten_patches_and_coords(patches, patch_coords)

    # Filter out background patches
    patch_mask = get_canny_foreground_mask(patches)
    patches = patches[patch_mask]
    patch_coords = patch_coords[patch_mask]

    # Cache the patches and patch coordinates
    patches, patch_coords = client.persist((patches, patch_coords))

    # Show a progress bar for loading and canny rejection
    tqdm_dask(patches, desc="Loading patches, filtering background")

    # We need deterministic chunk sizes for the following steps
    patches.compute_chunk_sizes()
    patch_coords.compute_chunk_sizes()
    patches = patches.rechunk((macenko_batch_size, patch_size, patch_size, 3))

    # Perform Macenko normalization
    normalizer = DaskMacenkoNormalizer(exact=True)
    normalizer.fit(stain_target)
    normalized = normalizer.normalize_flattened_image(patches.reshape(-1, 3))
    normalized = normalized.reshape(patches.shape)
    normalized = normalized.rechunk((feature_extraction_batch_size, patch_size, patch_size, 3))
    normalized = normalized.persist()
    tqdm_dask(normalized, desc="Macenko normalization")

    # Extract features
    features = extract_features(model, normalized, use_cuda=has_gpu_devices)

    # Cache the features
    # NOTE: if optimize_graph=True, then the graph optimization might move the model to a non-GPU worker (but
    # if we don't have any GPU workers, then this is not a problem)
    features = features.persist(optimize_graph=not has_gpu_devices)
    tqdm_dask(features, desc="Extracting features")

    # Save
    save_format = "h5py"
    if save_format == "h5py":
        logger.info("Saving features to HDF5")

        @dask.delayed
        def save_to_h5py(features, coords):
            import h5py

            with h5py.File("features.h5py", "w") as f:
                f.create_dataset("/features", data=features)
                f.create_dataset("/coords", data=coords)

        saving_future = client.submit(save_to_h5py, features, patch_coords)
    else:
        logger.info("Saving features to Zarr")
        saved_features = da.to_zarr(features.rechunk(-1), "features.zarr", overwrite=True, compute=False)
        saved_coords = da.to_zarr(patch_coords.rechunk(-1), "coords.zarr", overwrite=True, compute=False)
        saving_future = (saved_features, saved_coords)

    if synchronous_saving:
        tqdm_dask(saving_future, desc="Saving features")
    else:
        return saving_future


if __name__ == "__main__":
    t0 = time()

    # Setup the pipeline
    setup = setup_pipeline(gpu_devices=(2,))
    logger.info(f"Setup took {(t1:=time()) - t0:.2f} seconds")

    # Process the slide
    process_slide(setup, SLIDE_FILE, synchronous_saving=True)
    logger.info(f"Processing took {(t2:=time()) - t1:.2f} seconds")

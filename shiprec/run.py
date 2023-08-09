import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import dask.array as da
import dask
import torch
from dask.distributed import Client, wait, get_client, secede, rejoin
from loguru import logger
from concurrent.futures import Future
from torch import nn
from typing import NamedTuple, Sequence, Union, Tuple
from time import time
from omegaconf import DictConfig

import shiprec
from shiprec.slide import load_slide, Backend as SlideBackend
from shiprec.progress import tqdm_dask
from shiprec.cluster import MixedCluster
from shiprec.patching import split_slide_into_patches, flatten_patches_and_coords
from shiprec.canny import get_canny_foreground_mask
from shiprec.macenko.dask import DaskMacenkoNormalizer
from shiprec.feature_extraction import load_model, extract_features
from shiprec.utils import hydra, fix_dask_torch  # noqa


class PipelineSetup(NamedTuple):
    stain_target: da.Array
    model: Future[nn.Module]
    has_gpu_workers: bool


def find_slides(cfg: DictConfig) -> Sequence[Path]:
    """Find all slides in the input directory."""
    input_dir = Path(cfg.input.path)
    assert input_dir.is_dir(), f"Input directory {input_dir} does not exist"
    slide_files = list(input_dir.glob(cfg.input.glob))
    if slide_files:
        logger.info(f"Found {len(slide_files)} slides in {input_dir}")
    else:
        logger.warning(f"No slides found in {input_dir}")
    return slide_files


def setup(cfg: DictConfig) -> Tuple[Client, PipelineSetup]:
    start_time = time()
    # Start the Dask cluster
    cluster = MixedCluster(
        n_cpu_workers=cfg.cluster.n_cpu_workers,
        gpu_devices=cfg.cluster.gpu_devices or (),
        memory_limit=cfg.cluster.memory_limit,
    )
    client = Client(cluster.scheduler_address)
    gpu_workers = [
        (w.host, w.port) for w in cluster.workers.values() if w.resources is not None and w.resources.get("GPU", 0) > 0
    ]
    logger.info(f"Started Dask client with {len(cluster.workers)} workers, {len(gpu_workers)} of which are GPU workers")
    logger.info(f"Access the Dask dashboard at {cluster.dashboard_link}")

    # Load the stain target
    if cfg.pipeline.stain_normalization.enabled:
        assert cfg.pipeline.stain_normalization.method == "macenko", "Only Macenko normalization is supported"
        stain_target = cv2.cvtColor(cv2.imread(str(cfg.pipeline.stain_normalization.template)), cv2.COLOR_BGR2RGB)
        stain_target = da.from_array(stain_target)
        stain_target = client.persist(stain_target)
    else:
        assert (
            not cfg.output.normalized_patches.save
        ), "Cannot save normalized patches if stain normalization is disabled"
        stain_target = None

    # Load feature extraction model
    logger.info("Loading feature extraction model")
    model = load_model()
    model.eval()

    # Send the model to the workers
    # TODO: send the model to the GPU workers only (currently it is sent to all workers)
    dmodel = client.scatter(model)
    logger.info("Loaded and scattered feature extractor to workers")

    end_time = time()
    logger.info(f"Setup took {end_time - start_time:.2f} seconds")

    return client, PipelineSetup(
        stain_target=stain_target,
        model=dmodel,
        has_gpu_workers=len(cfg.cluster.gpu_devices or ()) > 0,
    )


def process_slide(
    pipeline_setup: PipelineSetup,
    slide_file: Path,
    cfg: DictConfig,
    client: Client = None,
    optimize_graph: bool = True,
):
    """Process a slide and save the features to disk."""
    start_time = time()

    if client is None:
        client = get_client()

    stain_target, model, has_gpu_devices = pipeline_setup

    # Load the slide
    slide = load_slide(slide_file, target_mpp=cfg.pipeline.target_mpp, backend=cfg.input.slide_backend)

    # Split the slide into patches
    patches, patch_coords = split_slide_into_patches(slide, patch_size=cfg.pipeline.patch_size)
    patches, patch_coords = flatten_patches_and_coords(patches, patch_coords)

    # Filter out background patches
    patch_mask = get_canny_foreground_mask(patches)
    patches = patches[patch_mask]
    patch_coords = patch_coords[patch_mask]

    # Cache the patches and patch coordinates
    patches, patch_coords = client.persist((patches, patch_coords), optimize_graph=optimize_graph)
    logger.debug("Loading patches, filtering background")
    wait((patches, patch_coords))
    # tqdm_dask((patches, patch_coords), desc="Loading patches, filtering background")

    # We need deterministic chunk sizes for the following steps
    patches.compute_chunk_sizes()
    patch_coords.compute_chunk_sizes()
    patches = patches.rechunk(
        (cfg.pipeline.stain_normalization.batch_size, cfg.pipeline.patch_size, cfg.pipeline.patch_size, 3)
    )

    # Perform Macenko normalization
    # TODO: do the fitting in the setup function
    normalizer = DaskMacenkoNormalizer(exact=True)
    normalizer.fit(stain_target)
    normalized = normalizer.normalize_flattened_image(patches.reshape(-1, 3))
    normalized = normalized.reshape(patches.shape)
    normalized = normalized.rechunk(
        (cfg.pipeline.feature_extraction.batch_size, cfg.pipeline.patch_size, cfg.pipeline.patch_size, 3)
    )
    normalized = normalized.persist(optimize_graph=optimize_graph)
    logger.debug("Performing Macenko normalization")
    wait(normalized)
    # tqdm_dask(normalized, desc="Macenko normalization")

    # Extract features
    features = extract_features(model, normalized, use_cuda=has_gpu_devices)

    # Cache the features
    # NOTE: if optimize_graph=True, then the graph optimization might move the model to a non-GPU worker (but
    # if we don't have any GPU workers, then this is not a problem)
    features = features.persist(optimize_graph=not has_gpu_devices)
    logger.debug("Extracting features")
    wait(features)
    # tqdm_dask(features, desc="Extracting features")

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

    # Show a progress bar for saving
    wait(saving_future)
    # tqdm_dask(saving_future, desc="Saving output")

    end_time = time()
    logger.info(f"Processing {slide_file} took {end_time - start_time:.2f} seconds")

    return slide_file


@hydra.main(config_path=str(Path(__file__).parent.parent), config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # Setup the pipeline
    client, pipeline_setup = setup(cfg)

    # Find all slides
    # slide_files = find_slides(cfg)
    SLIDE_FILE = "/data/data/TCGA-BRCA-DX-IMGS_1133/TCGA-AO-A12B-01Z-00-DX1.B215230B-5FF7-4B0A-9C1E-5F1658534B11.svs"
    slide_files = [SLIDE_FILE]

    # Process slides
    if cfg.pipeline.n_parallel_slides > 1:
        assert False
        futures = []
        for slide_file in slide_files:
            if len(futures) >= cfg.pipeline.n_parallel_slides:
                done, futures = wait(futures, return_when="FIRST_COMPLETED")
                for future in done:
                    logger.info(f"Finished processing {future.result()}")
            future = client.submit(process_slide, pipeline_setup, slide_file, cfg)
            futures = [future] + futures

        # Wait for all slides to finish
        logger.info(f"Waiting for {len(futures)} slides to finish processing")
        done, futures = wait(futures, return_when="ALL_COMPLETED")
    else:
        for slide_file in slide_files:
            process_slide(pipeline_setup, slide_file, cfg, client=client, optimize_graph=False)

    # Shutdown the Dask cluster
    logger.info("Shutting down Dask cluster")
    client.shutdown()

    logger.info("Done")


if __name__ == "__main__":
    main()

import shiprec  # must be first to register logger
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
from tqdm import tqdm
import functools

from shiprec.slide import load_slide, Backend as SlideBackend, MPPExtractionError
from shiprec.progress import tqdm_dask
from shiprec.cluster import MixedCluster
from shiprec.patching import split_slide_into_patches, flatten_patches_and_coords
from shiprec.canny import get_canny_foreground_mask
from shiprec.macenko.dask import DaskMacenkoNormalizer
from shiprec.feature_extraction import load_model, extract_features
from shiprec.utils import hydra, fix_dask_torch  # noqa


class PipelineSetup(NamedTuple):
    normalizer: DaskMacenkoNormalizer
    model: Future[nn.Module]
    has_gpu_workers: bool


def find_slides(cfg: DictConfig) -> Sequence[Path]:
    """Find all slides in the input directory."""
    input_dir = Path(cfg.input.path)
    assert input_dir.is_dir(), f"Input directory {input_dir} does not exist"
    slide_files = list(input_dir.glob(cfg.input.glob))
    if slide_files:
        logger.info(f"Found {len(slide_files)} slides in {input_dir}")
        status_folder = Path(cfg.output.path) / "status"
        slide_files = [
            slide_file for slide_file in slide_files if not (status_folder / f"{slide_file.stem}.done").exists()
        ]
        logger.info(f"{len(slide_files)} slides have not been processed yet")
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

    if cfg.pipeline.stain_normalization.enabled:
        # Load the stain target
        assert cfg.pipeline.stain_normalization.method == "macenko", "Only Macenko normalization is supported"
        stain_target = cv2.cvtColor(cv2.imread(str(cfg.pipeline.stain_normalization.template)), cv2.COLOR_BGR2RGB)
        stain_target = da.from_array(stain_target)
        # stain_target = client.persist(stain_target)

        # Load Macenko normalizer
        normalizer = DaskMacenkoNormalizer(exact=True)
        logger.debug("Fitting Macenko normalizer to template")
        normalizer.fit(stain_target, persist=True)
    else:
        assert (
            not cfg.output.normalized_patches.save
        ), "Cannot save normalized patches if stain normalization is disabled"
        normalizer = None

    # Load feature extraction model
    logger.debug("Loading feature extraction model")
    model = load_model(
        model=cfg.pipeline.feature_extraction.model, weights_path=Path(cfg.pipeline.feature_extraction.weights)
    )
    model.eval()

    # Send the model to the workers
    # TODO: send the model to the GPU workers only (currently it is sent to all workers)
    dmodel = client.scatter(model)
    logger.debug("Loaded and scattered feature extractor to workers")

    end_time = time()
    logger.info(f"Setup completed, took {end_time - start_time:.2f} seconds")

    return client, PipelineSetup(
        normalizer=normalizer,
        model=dmodel,
        has_gpu_workers=len(cfg.cluster.gpu_devices or ()) > 0,
    )


def secede_wait_rejoin(*args, **kwargs):
    secede()
    wait(*args, **kwargs)
    rejoin()


def process_slide(
    pipeline_setup: PipelineSetup,
    slide_file: Path,
    cfg: DictConfig,
    client: Client = None,
):
    """Process a slide and save the features to disk."""
    start_time = time()
    slide_file = Path(slide_file)
    if client is None:
        client = get_client()
        wait = secede_wait_rejoin

    optimize_graph = cfg.cluster.optimize_graph

    normalizer, model, has_gpu_devices = pipeline_setup

    # Check if the slide has already been processed
    status_file = Path(cfg.output.path) / "status" / f"{slide_file.stem}.done"
    if status_file.exists():
        logger.info(f"Skipping {slide_file.stem} because it has already been processed")
        return slide_file

    # Load the slide
    try:
        slide = load_slide(slide_file, target_mpp=cfg.pipeline.target_mpp, backend=cfg.input.slide_backend)
    except MPPExtractionError:
        logger.warning(f"Could not extract MPP for {slide_file}, skipping")
        return slide_file

    # Split the slide into patches
    patches, patch_coords = split_slide_into_patches(slide, patch_size=cfg.pipeline.patch_size)
    patch_grid_shape = patches.shape[:2]
    patches, patch_coords = flatten_patches_and_coords(patches, patch_coords)

    # Filter out background patches
    patch_mask = get_canny_foreground_mask(patches)
    patches = patches[patch_mask]
    patch_coords = patch_coords[patch_mask]

    # Cache the patches and patch coordinates
    patches, patch_coords, patch_mask = client.persist(
        (patches, patch_coords, patch_mask), optimize_graph=optimize_graph
    )
    logger.debug("Loading patches, filtering background")
    wait((patches, patch_coords))
    # tqdm_dask((patches, patch_coords), desc="Loading patches, filtering background")

    # We need deterministic chunk sizes for the following steps
    patches.compute_chunk_sizes()
    patch_coords.compute_chunk_sizes()
    patch_mask.compute_chunk_sizes()
    patches = patches.rechunk(
        (cfg.pipeline.stain_normalization.batch_size, cfg.pipeline.patch_size, cfg.pipeline.patch_size, 3)
    )
    logger.debug(f"Reduced number of patches from {np.prod(patch_grid_shape)} to {patches.shape[0]}")

    # Compute the patch index grid locally
    patch_grid = np.zeros(patch_grid_shape, dtype=int) - 1
    patch_grid = np.reshape(patch_grid, (-1,))
    computed_patch_mask = patch_mask.compute()  # should be cheap
    patch_grid[computed_patch_mask] = np.arange(computed_patch_mask.sum())
    patch_grid = np.reshape(patch_grid, patch_grid_shape)

    # Perform Macenko normalization
    if cfg.pipeline.stain_normalization.enabled:
        normalized = normalizer.normalize_flattened_image(patches.reshape(-1, 3))
        normalized = normalized.reshape(patches.shape)
        normalized = normalized.rechunk(
            (cfg.pipeline.feature_extraction.batch_size, cfg.pipeline.patch_size, cfg.pipeline.patch_size, 3)
        )
        normalized = normalized.persist(optimize_graph=optimize_graph)
        logger.debug("Performing Macenko normalization")
        wait(normalized)
        # tqdm_dask(normalized, desc="Macenko normalization")
    else:
        normalized = patches

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
    if cfg.output.format == "h5":
        logger.debug("Saving to HDF5")

        features_folder = Path(cfg.output.path) / "features"
        features_folder.mkdir(parents=True, exist_ok=True)
        cache_folder = Path(cfg.output.path) / "cache"
        cache_folder.mkdir(parents=True, exist_ok=True)

        @dask.delayed
        def save_to_h5py(features, coords):
            import h5py

            with h5py.File(features_folder / f"{slide_file.stem}.h5", "w") as f:
                if cfg.output.features.save:
                    f.create_dataset("/features", data=features.compute())
                if cfg.output.coords.save:
                    f.create_dataset("/coords", data=coords.compute())
                if cfg.patch_index_grid.save:
                    f.create_dataset("/patch_index_grid", data=patch_grid.compute())

        saving_future = client.submit(save_to_h5py, features, patch_coords)
    elif cfg.output.format == "zarr":
        logger.debug("Saving to Zarr")

        slide_folder = Path(cfg.output.path) / slide_file.stem

        saving_futures = []
        if cfg.output.features.save:
            saving_futures.append(
                da.to_zarr(features.rechunk(-1), slide_folder / "features.zarr", overwrite=True, compute=False)
            )
        if cfg.output.coords.save:
            saving_futures.append(
                da.to_zarr(patch_coords.rechunk(-1), slide_folder / "coords.zarr", overwrite=True, compute=False)
            )
        if cfg.output.patches.save:
            saving_futures.append(da.to_zarr(patches, slide_folder / "patches.zarr", overwrite=True, compute=False))
        if cfg.output.normalized_patches.save:
            saving_futures.append(
                da.to_zarr(normalized, slide_folder / "normalized_patches.zarr", overwrite=True, compute=False)
            )
        if cfg.output.patch_index_grid.save:
            saving_futures.append(
                da.to_zarr(
                    da.from_array(patch_grid).rechunk(-1),
                    slide_folder / "patch_index_grid.zarr",
                    overwrite=True,
                    compute=False,
                )
            )
        saving_future = tuple(saving_futures)
    else:
        raise ValueError(f"Unknown output format {cfg.output.format}")

    # Show a progress bar for saving
    wait(saving_future)
    # tqdm_dask(saving_future, desc="Saving output")

    # Write status file
    status_file.parent.mkdir(parents=True, exist_ok=True)
    with status_file.open("w") as f:
        f.write("done")

    end_time = time()
    logger.info(f"Processing {slide_file} took {end_time - start_time:.2f} seconds")

    return slide_file


def contextual_fn(fn, **context):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with logger.contextualize(**context):
            return fn(*args, **kwargs, **kwargs)

    return wrapper


@hydra.main(config_path=str(Path(__file__).parent.parent), config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    # Setup the pipeline
    client, pipeline_setup = setup(cfg)

    # Find all slides
    slide_files = find_slides(cfg)

    # Process slides
    t0 = time()
    if cfg.pipeline.n_parallel_slides > 1:
        futures = set()
        with tqdm(total=len(slide_files), desc="Processing slides") as pbar:
            for i, slide_file in enumerate(slide_files):
                if len(futures) >= cfg.pipeline.n_parallel_slides:
                    done, futures = wait(futures, return_when="FIRST_COMPLETED")
                    for future in done:
                        logger.info(f"Finished processing {future.result().stem}")
                        pbar.update(1)
                        del future
                logger.info(f"Queuing slide {i+1}/{len(slide_files)}: {slide_file.name}")
                future = client.submit(contextual_fn(process_slide, slide_id=i), pipeline_setup, slide_file, cfg)
                futures.add(future)

            # Wait for all slides to finish
            while futures:
                done, futures = wait(futures, return_when="FIRST_COMPLETED")
                for future in done:
                    logger.info(f"Finished processing {future.result().stem}")
                    pbar.update(1)
                    del future
            done, futures = wait(futures, return_when="ALL_COMPLETED")
    else:
        for slide_file in tqdm(slide_files, desc="Processing slides"):
            process_slide(pipeline_setup, slide_file, cfg, client=client)

    logger.info(f"Finished processing {len(slide_files)} slides in {time() - t0:.2f} seconds")

    # Shutdown the Dask cluster
    logger.info("Shutting down Dask cluster")
    client.shutdown()


if __name__ == "__main__":
    main()

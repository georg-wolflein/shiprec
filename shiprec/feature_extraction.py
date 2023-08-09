import hashlib
from shiprec.libs.ctranspath.swin_transformer import swin_tiny_patch4_window7_224, ConvStem
from torch import nn
import torch
from pathlib import Path
from torchvision import transforms as T
from concurrent.futures import Future
import dask.array as da
import dask
import numpy as np
from typing import Literal, Union
from loguru import logger

transform = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def load_model(model: Literal["ctranspath"], weights_path: Union[Path, str]) -> nn.Module:
    assert model == "ctranspath", "Only ctranspath is supported"
    weights_path = Path(weights_path)
    if not weights_path.exists():
        logger.info("Downloading weights")
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        import gdown

        gdown.download(
            "https://drive.google.com/u/0/uc?id=1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX&export=download",
            str(weights_path),
            quiet=False,
        )

    sha256 = hashlib.sha256()
    with weights_path.open("rb") as f:
        while True:
            data = f.read(1 << 16)
            if not data:
                break
            sha256.update(data)

    assert sha256.hexdigest() == "7c998680060c8743551a412583fac689db43cec07053b72dfec6dcd810113539"

    model = swin_tiny_patch4_window7_224(embed_layer=ConvStem, pretrained=False)
    model.head = nn.Identity()

    ctranspath = torch.load(weights_path, map_location=torch.device("cpu"))
    model.load_state_dict(ctranspath["model"], strict=True)

    model.eval()
    return model


def extract_features(model: Future[nn.Module], patches: da.Array, use_cuda: bool = True) -> da.Array:
    # patches is of shape (N, 224, 224, 3)

    device = "cuda" if use_cuda else "cpu"

    def predict(batch: np.ndarray, model) -> np.ndarray:
        model.to(device)  # will only be slow the first time per worker (because we leave the model on the GPU)
        batch = torch.from_numpy(batch).float() / 255.0
        batch = transform(batch).to(device)
        # batch is of shape (batch_size, 3, 224, 224)
        with torch.no_grad():
            features = model(batch).cpu().numpy()
        model.cpu()  # need to do this otherwise we get a serialization error
        return features

    # Move channel dimension to PyTorch's expected position
    tensor_patches = da.moveaxis(patches, -1, 1)  # (N, 3, 224, 224)

    # Extract features
    with dask.annotate(resources={"GPU": 1} if use_cuda else {}):
        features = da.map_blocks(
            predict,
            tensor_patches,
            model,
            meta=np.zeros((), dtype=float),
            drop_axis=(1, 2, 3),
            new_axis=1,
            chunks=(tensor_patches.chunks[0], 768),
        )

    return features

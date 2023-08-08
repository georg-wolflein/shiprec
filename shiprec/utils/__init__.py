import distributed.protocol.torch  # noqa
from distributed.protocol.serialize import serialize, deserialize, dask_serialize
import torch


@dask_serialize.register(torch.Tensor)
def serialize_torch_Tensor(t):
    """Need to fix this implementation when gpu is on device

    This is a bug in dask

    copied from here: https://github.com/dask/distributed/blob/172f23d78ac1f8c6117b9edfd0019ec94cd7d39d/distributed/protocol/torch.py#L15
    """  # noqa
    requires_grad_ = t.requires_grad

    if requires_grad_:
        sub_header, frames = serialize(t.detach().cpu().numpy())
    else:
        sub_header, frames = serialize(t.cpu().numpy())

    header = {"sub-header": sub_header}
    if t.grad is not None:
        grad_header, grad_frames = serialize(t.grad.numpy())
        header["grad"] = {"header": grad_header, "start": len(frames)}
        frames += grad_frames
    header["requires_grad"] = requires_grad_
    header["device"] = t.device.type
    return header, frames

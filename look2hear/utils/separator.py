###
# Author: Kai Li
# Date: 2021-06-18 16:32:50
# LastEditors: Kai Li
# LastEditTime: 2021-06-19 01:02:04
###
import os
import warnings
import torch
import numpy as np
import soundfile as sf


def get_device(tensor_or_module, default=None):
    if hasattr(tensor_or_module, "device"):
        return tensor_or_module.device
    elif hasattr(tensor_or_module, "parameters"):
        return next(tensor_or_module.parameters()).device
    elif default is None:
        raise TypeError(
            f"Don't know how to get device of {type(tensor_or_module)} object"
        )
    else:
        return torch.device(default)


class Separator:
    def forward_wav(self, wav, **kwargs):
        raise NotImplementedError

    def sample_rate(self):
        raise NotImplementedError


def separate(model, wav, **kwargs):
    if isinstance(wav, np.ndarray):
        return numpy_separate(model, wav, **kwargs)
    elif isinstance(wav, torch.Tensor):
        return torch_separate(model, wav, **kwargs)
    else:
        raise ValueError(
            f"Only support filenames, numpy arrays and torch tensors, received {type(wav)}"
        )


@torch.no_grad()
def torch_separate(model: Separator, wav: torch.Tensor, **kwargs) -> torch.Tensor:
    """Core logic of `separate`."""
    if model.in_channels is not None and wav.shape[-2] != model.in_channels:
        raise RuntimeError(
            f"Model supports {model.in_channels}-channel inputs but found audio with {wav.shape[-2]} channels."
            f"Please match the number of channels."
        )
    # Handle device placement
    input_device = get_device(wav, default="cpu")
    model_device = get_device(model, default="cpu")
    wav = wav.to(model_device)
    # Forward
    separate_func = getattr(model, "forward_wav", model)
    out_wavs = separate_func(wav, **kwargs)

    # FIXME: for now this is the best we can do.
    out_wavs *= wav.abs().sum() / (out_wavs.abs().sum())

    # Back to input device (and numpy if necessary)
    out_wavs = out_wavs.to(input_device)
    return out_wavs


def numpy_separate(model: Separator, wav: np.ndarray, **kwargs) -> np.ndarray:
    """Numpy interface to `separate`."""
    wav = torch.from_numpy(wav)
    out_wavs = torch_separate(model, wav, **kwargs)
    out_wavs = out_wavs.data.numpy()
    return out_wavs

###
# Author: Kai Li
# Date: 2021-06-17 23:08:32
# LastEditors: Please set LastEditors
# LastEditTime: 2022-05-26 18:06:22
###
import os
import torch
import torch.nn as nn
from typing import Union, Dict, List
from functools import lru_cache
from hashlib import sha256
import requests
from torch import hub
import huggingface_hub


CACHE_DIR = os.path.expanduser("~/.cache/torch/tdanet")

def cached_download(filename_or_url):
    if os.path.isfile(filename_or_url):
        return filename_or_url

    if filename_or_url.startswith(huggingface_hub.HUGGINGFACE_CO_URL_HOME):
        filename_or_url = filename_or_url[len(huggingface_hub.HUGGINGFACE_CO_URL_HOME) :]

    if filename_or_url.startswith(("http://", "https://")):
        url = filename_or_url
    else:
        # Finally, let's try to find it on Hugging Face model hub
        # e.g. julien-c/DPRNNTasNet-ks16_WHAM_sepclean is a valid model id
        # and  julien-c/DPRNNTasNet-ks16_WHAM_sepclean@main supports specifying a commit/branch/tag.
        if "@" in filename_or_url:
            model_id = filename_or_url.split("@")[0]
            revision = filename_or_url.split("@")[1]
        else:
            model_id = filename_or_url
            revision = None
        return huggingface_hub.hf_hub_download(
            repo_id=model_id,
            filename=huggingface_hub.PYTORCH_WEIGHTS_NAME,
            cache_dir=get_cache_dir(),
            revision=revision,
        )

    cached_filename = url_to_filename(url)
    cached_dir = os.path.join(get_cache_dir(), cached_filename)
    cached_path = os.path.join(cached_dir, "model.pth")

    os.makedirs(cached_dir, exist_ok=True)
    if not os.path.isfile(cached_path):
        hub.download_url_to_file(url, cached_path)
        return cached_path
    # It was already downloaded
    print(f"Using cached model `{filename_or_url}`")
    return cached_path


def url_to_filename(url):
    """Consistently convert ``url`` into a filename."""
    _bytes = url.encode("utf-8")
    _hash = sha256(_bytes)
    filename = _hash.hexdigest()
    return filename


def get_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)
    return CACHE_DIR


@lru_cache()
def model_list(
    endpoint=huggingface_hub.HUGGINGFACE_CO_URL_HOME, name_only=False
) -> Union[str, List[Dict]]:
    """Get the public list of all the models on huggingface with an 'asteroid' tag."""
    path = "{}api/models?full=true&filter=asteroid".format(endpoint)
    r = requests.get(path)
    r.raise_for_status()
    all_models = r.json()
    if name_only:
        return [x["modelId"] for x in all_models]
    return all_models

def _unsqueeze_to_3d(x):
    """Normalize shape of `x` to [batch, n_chan, time]."""
    if x.ndim == 1:
        return x.reshape(1, 1, -1)
    elif x.ndim == 2:
        return x.unsqueeze(1)
    else:
        return x


def pad_to_appropriate_length(x, lcm):
    values_to_pad = int(x.shape[-1]) % lcm
    if values_to_pad:
        appropriate_shape = x.shape
        padded_x = torch.zeros(
            list(appropriate_shape[:-1])
            + [appropriate_shape[-1] + lcm - values_to_pad],
            dtype=torch.float32,
        ).to(x.device)
        padded_x[..., : x.shape[-1]] = x
        return padded_x
    return x


class BaseModel(nn.Module):
    def __init__(self, sample_rate, in_chan=1):
        super().__init__()
        self._sample_rate = sample_rate
        self._in_chan = in_chan

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def sample_rate(self,):
        return self._sample_rate

    @staticmethod
    def load_state_dict_in_audio(model, pretrained_dict):
        model_dict = model.state_dict()
        update_dict = {}
        for k, v in pretrained_dict.items():
            if "audio_model" in k:
                update_dict[k[12:]] = v
        model_dict.update(update_dict)
        model.load_state_dict(model_dict)
        return model

    
    @staticmethod
    def from_pretrain(pretrained_model_conf_or_path, *args, **kwargs):
        from . import get
        if os.path.exists(pretrained_model_conf_or_path):
            conf = torch.load(
                pretrained_model_conf_or_path, map_location="cpu"
            )  # Attempt to find the model and instantiate it.
            model_class = get(conf["model_name"])
            # model_class = get("Conv_TasNet")
            model = model_class(*args, **kwargs)
            model.load_state_dict(conf["state_dict"])
            return model
        else:
            cached_model = cached_download(pretrained_model_conf_or_path)
            conf = torch.load(cached_model, map_location="cpu")
            model_class = get(conf["model_name"])
            # model_class = get("Conv_TasNet")
            model = model_class(*args, **conf["model_args"])
            model.load_state_dict(conf["state_dict"])
            return model
        

    def serialize(self):
        import pytorch_lightning as pl  # Not used in torch.hub

        model_conf = dict(
            model_name=self.__class__.__name__,
            state_dict=self.get_state_dict(),
            model_args=self.get_model_args(),
        )
        # Additional infos
        infos = dict()
        infos["software_versions"] = dict(
            torch_version=torch.__version__, pytorch_lightning_version=pl.__version__,
        )
        model_conf["infos"] = infos
        return model_conf

    def get_state_dict(self):
        """In case the state dict needs to be modified before sharing the model."""
        return self.state_dict()

    def get_model_args(self):
        """Should return args to re-instantiate the class."""
        raise NotImplementedError

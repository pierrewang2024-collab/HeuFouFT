# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Any, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge


class FourierFTLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("fourierft_spectrum",)
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("fourierft_n_frequency", "fourierft_scaling", "fourierft_random_loc_seed")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.fourierft_n_frequency = {}
        self.fourierft_scaling = {}
        self.fourierft_spectrum = nn.ParameterDict({})
        self.indices = {}
        self.fourierft_random_loc_seed = {}
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            self.in_features, self.out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

    def update_layer(self, adapter_name, n_frequency, scaling, init_weights, random_loc_seed,
                     sampling="uniform", bandpass_fc=0.0, bandpass_width=20.0,
                     sampling_indices_file=None, sampling_seed=None,
                     module_name=None):
        if n_frequency <= 0:
            raise ValueError(f"`n_frequency` should be a positive integer value but the value passed is {n_frequency}")
        if n_frequency > self.in_features * self.out_features:
            raise ValueError(
                f"`n_frequency` should be less than or equal to the product of the input and output dimensions "
                f"but the value passed is {n_frequency} and the product is {self.in_features * self.out_features}"
            )
        self.fourierft_n_frequency[adapter_name] = n_frequency
        self.fourierft_random_loc_seed[adapter_name] = random_loc_seed
        # Coordinate sampling strategy (uniform / bandpass / fixed indices).
        # Imported lazily to avoid a circular import at module load time.
        from .sampling import make_indices

        seed = random_loc_seed if sampling_seed is None else sampling_seed
        rng = np.random.default_rng(seed)
        idx = make_indices(
            sampling,
            self.out_features,
            self.in_features,
            n_frequency,
            rng,
            fc=bandpass_fc,
            width=bandpass_width,
            indices_file=sampling_indices_file,
            module_name=module_name,
        )
        if idx.shape[1] != n_frequency:
            raise ValueError(
                f"Loaded indices contain {idx.shape[1]} coordinates but n_frequency is {n_frequency}"
            )
        if int(idx[0].max()) >= self.out_features or int(idx[1].max()) >= self.in_features:
            raise ValueError(
                "Loaded indices exceed the layer spectrum shape "
                f"({self.out_features}, {self.in_features})"
            )
        self.indices[adapter_name] = idx
        self.fourierft_scaling[adapter_name] = scaling
        # Actual trainable parameters
        self.fourierft_spectrum[adapter_name] = nn.Parameter(torch.randn(n_frequency), requires_grad=True)

        if init_weights:
            self.reset_fourier_parameters(adapter_name)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    @torch.no_grad()
    def reset_fourier_parameters(self, adapter_name):
        if adapter_name in self.fourierft_spectrum.keys():
            nn.init.zeros_(self.fourierft_spectrum[adapter_name])

    def get_delta_weight(self, adapter) -> torch.Tensor:
        spectrum = self.fourierft_spectrum[adapter]
        indices = self.indices[adapter].to(spectrum.device)
        dense_spectrum = torch.zeros(self.out_features, self.in_features, device=spectrum.device, dtype=spectrum.dtype)
        dense_spectrum[indices[0, :], indices[1, :]] = spectrum
        delta_weight = torch.fft.ifft2(dense_spectrum).real * self.fourierft_scaling[adapter]
        return delta_weight


class FourierFTLinear(nn.Module, FourierFTLayer):
    # FourierFT implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        n_frequency: int = 1000,
        scaling: float = 150.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        init_weights: Union[bool, str] = False,
        random_loc_seed: int = 777,
        sampling: str = "uniform",
        bandpass_fc: float = 0.0,
        bandpass_width: float = 20.0,
        sampling_indices_file: Optional[str] = None,
        sampling_seed: Optional[int] = None,
        module_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        FourierFTLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name, n_frequency, scaling, init_weights, random_loc_seed,
            sampling=sampling, bandpass_fc=bandpass_fc,
            bandpass_width=bandpass_width,
            sampling_indices_file=sampling_indices_file,
            sampling_seed=sampling_seed,
            module_name=module_name,
        )

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.fourierft_spectrum.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.fourierft_spectrum.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        return super().get_delta_weight(adapter)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.fourierft_spectrum.keys():
                    continue

                delta_w = self.get_delta_weight(active_adapter)
                x = x.to(delta_w.dtype)
                result = result + F.linear(x, delta_w)

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "fourierft." + rep

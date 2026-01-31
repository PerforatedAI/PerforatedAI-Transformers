# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import importlib
from typing import TYPE_CHECKING, Optional, Union

from packaging import version

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from ..utils import (
    ACCELERATE_MIN_VERSION,
    BITSANDBYTES_MIN_VERSION,
    is_accelerate_available,
    is_bitsandbytes_available,
    is_torch_available,
    is_torch_hpu_available,
    is_torch_npu_available,
    is_torch_xpu_available,
    logging,
)
from .quantizers_utils import get_module_from_name


if is_torch_available():
    import torch

    from ..core_model_loading import WeightConverter

logger = logging.get_logger(__name__)


class Bnb8BitHfQuantizer(HfQuantizer):
    """
    8-bit quantization from bitsandbytes quantization method
    """

    requires_calibration = False

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, *args, **kwargs):
        if not is_accelerate_available():
            raise ImportError(
                f"Using `bitsandbytes` 8-bit quantization requires accelerate: `pip install 'accelerate>={ACCELERATE_MIN_VERSION}'`"
            )
        if not is_bitsandbytes_available():
            raise ImportError(
                f"Using `bitsandbytes` 8-bit quantization requires bitsandbytes: `pip install -U bitsandbytes>={BITSANDBYTES_MIN_VERSION}`"
            )

        from ..integrations import validate_bnb_backend_availability

        validate_bnb_backend_availability(raise_exception=True)

        device_map = kwargs.get("device_map")
        if not self.quantization_config.llm_int8_enable_fp32_cpu_offload and isinstance(device_map, dict):
            values = set(device_map.values())
            if values != {"cpu"} and ("cpu" in values or "disk" in values):
                raise ValueError(
                    "Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit the "
                    "quantized model. If you want to dispatch the model on the CPU or the disk while keeping these modules "
                    "in 32-bit, you need to set `llm_int8_enable_fp32_cpu_offload=True` and pass a custom `device_map` to "
                    "`from_pretrained`. Check "
                    "https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu "
                    "for more details. "
                )

    def adjust_max_memory(self, max_memory: dict[str, int | str]) -> dict[str, int | str]:
        # need more space for buffers that are created during quantization
        max_memory = {key: val * 0.90 for key, val in max_memory.items()}
        return max_memory

    def update_device_map(self, device_map):
        if device_map is None:
            if torch.cuda.is_available():
                device_map = {"": torch.cuda.current_device()}
            elif is_torch_npu_available():
                device_map = {"": f"npu:{torch.npu.current_device()}"}
            elif is_torch_hpu_available():
                device_map = {"": f"hpu:{torch.hpu.current_device()}"}
            elif is_torch_xpu_available():
                device_map = {"": torch.xpu.current_device()}
            else:
                device_map = {"": "cpu"}
            logger.info(
                "The device_map was not initialized. "
                f"Setting device_map to {device_map}. "
                "If you want to use the model for inference, please set device_map ='auto' "
            )
        return device_map

    def param_element_size(self, model: "PreTrainedModel", param_name: str, param: "torch.Tensor") -> float:
        "Return the element size (in bytes) for `param_name`."
        if self.param_needs_quantization(model, param_name):
            # 8-bit
            return 1
        return super().param_element_size(model, param_name, param)

    def update_unexpected_keys(self, model, unexpected_keys: list[str]) -> list[str]:
        bnb_keys = ["SCB", "weight_format"]
        return [k for k in unexpected_keys if not any(k.endswith(x) for x in bnb_keys)]

    def param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        import bitsandbytes as bnb

        module, name = get_module_from_name(model, param_name)
        return isinstance(module, bnb.nn.Linear8bitLt) and name != "bias"

    def create_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        **kwargs,
    ):
        import bitsandbytes as bnb

        module, tensor_name = get_module_from_name(model, param_name)

        if self.pre_quantized and not self.is_serializable():
            raise ValueError(
                "Detected int8 weights but the version of bitsandbytes is not compatible with int8 serialization. "
                "Make sure to download the latest `bitsandbytes` version. `pip install --upgrade bitsandbytes`."
            )
        # Those 2 can only happen when self.pre_quantized == True
        if tensor_name == "SCB":
            setattr(module.weight, "SCB", param_value.to(target_device))
            return
        # It's not used, but it's getting serialized for BC reason...
        elif tensor_name == "weight_format":
            return

        # Support models using `Conv1D` in place of `nn.Linear` (e.g. openai-community/gpt2) by transposing the weight matrix prior to quantization.
        # Since weights are saved in the correct "orientation", we skip transposing when loading.
        if issubclass(module.source_cls, Conv1D) and not self.pre_quantized:
            param_value = param_value.T

        old_value = getattr(module, tensor_name)
        kwargs = old_value.__dict__
        kwargs.pop("_is_hf_initialized", None)
        # Need to pop SCB and reset it because of bnb internals that modifies its value when switching devices ...
        SCB = kwargs.pop("SCB", None)
        new_value = bnb.nn.Int8Params(param_value.to("cpu"), requires_grad=False, **kwargs).to(target_device)
        if SCB is not None:
            setattr(new_value, "SCB", SCB)
        # Set it to the module
        module._parameters[tensor_name] = new_value

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        model.is_loaded_in_8bit = True
        model.is_8bit_serializable = self.is_serializable()
        return model

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        device_map,
        **kwargs,
    ):
        from ..integrations import replace_with_bnb_linear

        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.llm_int8_skip_modules, model._keep_in_fp32_modules
        )

        if self.quantization_config.llm_int8_enable_fp32_cpu_offload:
            if isinstance(device_map, dict):
                keys_on_cpu = [key for key, value in device_map.items() if value in ["disk", "cpu"]]
                self.modules_to_not_convert.extend(keys_on_cpu)

        model = replace_with_bnb_linear(
            model, modules_to_not_convert=self.modules_to_not_convert, quantization_config=self.quantization_config
        )

        model.config.quantization_config = self.quantization_config

    def is_serializable(self, safe_serialization=None):
        _bnb_supports_8bit_serialization = version.parse(importlib.metadata.version("bitsandbytes")) > version.parse(
            "0.37.2"
        )

    def is_serializable(self):
        return True

    @property
    def is_trainable(self) -> bool:
        return True

    def _dequantize(self, model, dtype=None):
        from ..integrations import dequantize_and_replace

        model = dequantize_and_replace(model, quantization_config=self.quantization_config, dtype=dtype)
        return model

    def get_quantize_ops(self):
        from ..integrations.bitsandbytes import Bnb8bitQuantize

        return Bnb8bitQuantize(self)

    def get_weight_conversions(self):
        from ..integrations.bitsandbytes import Bnb8bitDeserialize

        if self.pre_quantized:
            return [
                WeightConverter(
                    source_patterns=["SCB", "weight_format", "weight"],
                    target_patterns="weight",
                    operations=[Bnb8bitDeserialize(self)],
                )
            ]
        return []

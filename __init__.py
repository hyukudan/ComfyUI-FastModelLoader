"""
ComfyUI-FastModelLoader: High-Speed Streaming Safetensors Loader & Memory Resilience Patcher
Copyright (c) 2026 hyukudan
Licensed under the MIT License
"""

import logging
import torch
import safetensors.torch
import comfy.utils
import comfy.model_patcher

logger = logging.getLogger("ComfyUI-FastModelLoader")

# 1. Update-Proof Core Safetensors Direct I/O Patcher
_original_load_torch_file = comfy.utils.load_torch_file

def _fast_load_torch_file(ckpt, safe_load=False, device=None, return_metadata=False):
    if device is None:
        device = torch.device("cpu")
    
    is_safetensors = ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft")
    aimdo_on = getattr(getattr(comfy, 'memory_management', None), 'aimdo_enabled', False)
    
    if is_safetensors and not aimdo_on:
        metadata = None
        if return_metadata:
            try:
                with safetensors.safe_open(ckpt, framework="pt", device="cpu") as f:
                    metadata = f.metadata()
            except Exception:
                metadata = None
        try:
            # Fast direct streaming I/O bypasses Windows mmap page fault crashes (0xc0000005)
            sd = safetensors.torch.load_file(ckpt, device=device.type, backend="pread")
            if return_metadata:
                return sd, metadata
            return sd
        except Exception:
            pass
            
    return _original_load_torch_file(ckpt, safe_load=safe_load, device=device, return_metadata=return_metadata)

comfy.utils.load_torch_file = _fast_load_torch_file

# 2. ModelPatcher & Quantized Memory Offload GC Resilience Patch
_original_detach = comfy.model_patcher.ModelPatcher.detach

def _safe_detach(self, unpatch_all=True):
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    self.eject_model()
    self.model_patches_to(self.offload_device)
    if unpatch_all:
        try:
            self.unpatch_model(self.offload_device, unpatch_weights=unpatch_all)
        except Exception:
            pass
    callbacks_mp = getattr(comfy.model_patcher, 'CallbacksMP', None)
    if callbacks_mp is not None:
        for callback in self.get_all_callbacks(callbacks_mp.ON_DETACH):
            try:
                callback(self, unpatch_all)
            except Exception:
                pass
    return self.model

comfy.model_patcher.ModelPatcher.detach = _safe_detach

_original_del = getattr(comfy.model_patcher.ModelPatcher, '__del__', None)

def _safe_del(self):
    try:
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        self.unpin_all_weights()
        self.detach(unpatch_all=False)
    except Exception:
        pass

comfy.model_patcher.ModelPatcher.__del__ = _safe_del

# Safe Quantized Module Device Transfer (prevents Windows access violations when offloading INT8/FP8 weights)
try:
    import comfy.ops
    _orig_quant_apply = getattr(comfy.ops, '_quantized_apply', None)
    if _orig_quant_apply is not None:
        def _safe_quantized_apply(module, fn, recurse=True):
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            if recurse:
                for child in module.children():
                    child._apply(fn)
            for key, param in module._parameters.items():
                if param is None:
                    continue
                try:
                    p = fn(param)
                except Exception:
                    try:
                        p = param.detach().clone().cpu()
                    except Exception:
                        p = param
                if (not torch.is_inference_mode_enabled()) and p.is_inference():
                    p = p.clone()
                module.register_parameter(key, torch.nn.Parameter(p, requires_grad=False))
            for key, buf in module._buffers.items():
                if buf is not None:
                    try:
                        module._buffers[key] = fn(buf)
                    except Exception:
                        pass
            return module

        comfy.ops._quantized_apply = _safe_quantized_apply
except Exception:
    pass

# 3. Safe Video Batch Resizer (prevents numpy._ArrayMemoryError when scaling video batches at 2K)
_original_lanczos = getattr(comfy.utils, 'lanczos', None)

def _safe_lanczos(samples, width, height):
    if samples.shape[0] > 8 or (samples.shape[-1] * samples.shape[-2] > 1024 * 1024):
        s = samples if samples.shape[1] in (1, 3, 4) else samples.movedim(-1, 1)
        out = torch.nn.functional.interpolate(s, size=(height, width), mode="bicubic", align_corners=False)
        if samples.shape[1] not in (1, 3, 4):
            out = out.movedim(1, -1)
        return out
    try:
        return _original_lanczos(samples, width, height)
    except Exception:
        s = samples if samples.shape[1] in (1, 3, 4) else samples.movedim(-1, 1)
        out = torch.nn.functional.interpolate(s, size=(height, width), mode="bicubic", align_corners=False)
        if samples.shape[1] not in (1, 3, 4):
            out = out.movedim(1, -1)
        return out

if _original_lanczos is not None:
    comfy.utils.lanczos = _safe_lanczos

# 4. Update-Proof Model Architecture Detection Patch (detects LTX 2.5 22B cross_attention_adaln & audio channels)
try:
    import comfy.model_detection
    _orig_detect_unet_config = getattr(comfy.model_detection, 'detect_unet_config', None)
    if _orig_detect_unet_config is not None:
        def _safe_detect_unet_config(state_dict, key_prefix, metadata=None):
            dit_config = _orig_detect_unet_config(state_dict, key_prefix, metadata=metadata)
            if dit_config is not None and isinstance(dit_config, dict):
                if dit_config.get("image_model") in ("ltxv", "ltxav"):
                    adaln_bias_key = f"{key_prefix}adaln_single.linear.bias"
                    if adaln_bias_key in state_dict:
                        bias_len = state_dict[adaln_bias_key].shape[0]
                        dit_config["cross_attention_adaln"] = (bias_len == 36864 or bias_len > 24576)

                    conn_key = f"{key_prefix}audio_embeddings_connector.learnable_registers"
                    if conn_key in state_dict:
                        dit_config["caption_channels"] = state_dict[conn_key].shape[-1]
            return dit_config

        comfy.model_detection.detect_unet_config = _safe_detect_unet_config
except Exception:
    pass

logger.info("[ComfyUI-FastModelLoader] Direct pread I/O streaming, GC resilience & Safe Video Resizer active.")

try:
    from .fast_loader_nodes import FastModelLoader, FastDiffusionModelLoader

    NODE_CLASS_MAPPINGS = {
        "FastModelLoader": FastModelLoader,
        "FastDiffusionModelLoader": FastDiffusionModelLoader,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "FastModelLoader": "⚡ Fast Model Checkpoint Loader (Pread)",
        "FastDiffusionModelLoader": "⚡ Fast Diffusion Model Loader (Pread)",
    }
except Exception as e:
    logger.warning(f"Could not import node classes: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

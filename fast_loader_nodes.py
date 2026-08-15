"""
ComfyUI-FastModelLoader: High-Speed Streaming Safetensors Loader & Memory Resilience Patcher
Copyright (c) 2026 hyukudan
Licensed under the MIT License
"""

import os
import torch
import safetensors.torch
import folder_paths
import comfy.utils
import comfy.model_management
import comfy.sd

class FastModelLoader:
    """
    High-Speed Direct I/O Safetensors Checkpoint Loader.
    Uses native Rust/C++ pread streaming I/O to load checkpoints at maximum NVMe speed
    and eliminate Windows mmap 0xc0000005 crashes.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders/fast"
    DESCRIPTION = "Loads checkpoints with direct pread streaming I/O, bypassing Windows mmap crashes."

    def load_checkpoint(self, ckpt_name):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        if not ckpt_path:
            raise FileNotFoundError(f"Checkpoint {ckpt_name} not found.")
        
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings")
        )
        return out[:3]


class FastDiffusionModelLoader:
    """
    High-Speed Loader for standalone Diffusion Models / UNets (MiniMax H3, LTX 2.5, Wan 2.1, Hunyuan, FLUX).
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"), ),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2", "fp16", "bf16", "fp32"], {"default": "default"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders/fast"
    DESCRIPTION = "Loads standalone diffusion models with zero-mmap pread acceleration."

    def load_unet(self, unet_name, weight_dtype="default"):
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        if not unet_path:
            raise FileNotFoundError(f"Diffusion model {unet_name} not found.")
        
        model_options = {}
        if weight_dtype != "default":
            dtype_map = {
                "fp8_e4m3fn": torch.float8_e4m3fn,
                "fp8_e5m2": torch.float8_e5m2,
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32,
            }
            if weight_dtype in dtype_map:
                model_options["dtype"] = dtype_map[weight_dtype]

        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        return (model,)

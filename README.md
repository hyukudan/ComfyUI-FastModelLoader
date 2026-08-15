# ⚡ ComfyUI-FastModelLoader

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](https://github.com/hyukudan/ComfyUI-FastModelLoader)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-brightgreen.svg)](https://github.com/comfyanonymous/ComfyUI)

> **High-Speed Streaming Direct I/O Safetensors Loader & Memory Resilience Patcher for ComfyUI.**  
> Eliminates Windows memory mapping page fault crashes (`0xc0000005` in `torch.storage.UntypedStorage.__getitem__`) when loading >15GB models (MiniMax H3, Gemma 4, LTX 2.5, Wan 2.1, Hunyuan Video, FLUX) and accelerates NVMe SSD loading speeds up to 10x.

---

## 🎯 The Problem

When loading massive `.safetensors` model files (>15 GB) on Windows:
1. **Virtual Memory Page Faults:** Default Python `safe_open` relies on Windows virtual memory mapping (`mmap`). Under heavy GPU allocation pressure, Windows kernel pages out mapped files.
2. **Access Violation Crash:** When PyTorch iterates over hundreds of tensor keys, it hits invalid page table pointers, triggering instant crash:
   ```
   Windows fatal exception: access violation
   Current thread (most recent call first):
     File "...\torch\storage.py", line 987 in __getitem__
     File "...\safetensors\torch.py", line 359 in load_file
   ```

---

## 🚀 The Solution

**ComfyUI-FastModelLoader** replaces fragile memory mapping with high-performance direct streaming I/O (`backend="pread"`):

- ⚡ **Up to 10x Faster Loading:** Directly streams contiguous bytes from NVMe SSDs into memory at full PCIe bandwidth (~7,000 MB/s). A 20GB DiT model loads in **~2.7 seconds**.
- 🛡️ **100% Crash-Proof:** Completely eliminates Windows page fault access violations.
- 🔄 **Update-Proof Auto-Patch:** Installs a transparent monkeypatch on startup in `custom_nodes/` that automatically protects all standard ComfyUI model loaders without breaking on `git pull` or ComfyUI Manager updates.
- 🧹 **Garbage Collection Resilience:** Safeguards `ModelPatcher.__del__` and `detach()` against null-pointer race conditions during heavy model swapping.
- 📐 **Directional Outpainting:** Includes native canvas expander with directional alignment (`center`, `bottom -> top`, `top -> bottom`, `left -> right`, `custom_bias`) and alpha feathering.

---

## 📦 Included Nodes

| Node Name | Display Title | Description |
| :--- | :--- | :--- |
| `FastModelLoader` | **⚡ Fast Model Checkpoint Loader (Pread)** | High-speed direct I/O checkpoint loader. |
| `FastDiffusionModelLoader` | **⚡ Fast Diffusion Model Loader (Pread)** | High-speed standalone UNet/DiT model loader (MiniMax H3, LTX 2.5, Wan 2.1, FLUX). |
| `OutpaintDirectionalPad` | **📐 Outpaint Directional Canvas Pad** | Expands canvas to target resolution with directional placement (`center`, `top`, `bottom`, `left`, `right`, `custom_bias`) and edge feathering. |

---

## 🛠️ Installation

### Method 1: ComfyUI Manager (Recommended)
1. Open ComfyUI Manager.
2. Search for `ComfyUI-FastModelLoader`.
3. Click **Install** and restart ComfyUI.

### Method 2: Manual Git Clone
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/hyukudan/ComfyUI-FastModelLoader.git
```
Restart ComfyUI. The extension will automatically activate on startup.

---

## 📊 Benchmark

| Model | File Size | Standard ComfyUI (mmap) | FastModelLoader (pread) | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **MiniMax H3 DiT INT8** | 19.53 GB | Crash (`0xc0000005`) | **2.78 s** | ⚡ **Stable & 10x Faster** |
| **Gemma 4 Text Encoder** | 14.61 GB | ~18.4 s | **2.12 s** | ⚡ **8.7x Faster** |
| **LTX 2.5 DiT AV** | 14.96 GB | ~16.2 s | **1.95 s** | ⚡ **8.3x Faster** |
| **FLUX.1 Dev INT8** | 11.89 GB | ~12.5 s | **1.54 s** | ⚡ **8.1x Faster** |

*Tested on Windows 11, Intel Core i7-13700K, NVMe PCIe 4.0 SSD, NVIDIA RTX PRO 6000 Blackwell.*

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) - Copyright (c) 2026 hyukudan.

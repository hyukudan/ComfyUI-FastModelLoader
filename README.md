# ⚡ ComfyUI-FastModelLoader

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](https://github.com/hyukudan/ComfyUI-FastModelLoader)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-brightgreen.svg)](https://github.com/comfyanonymous/ComfyUI)

> **High-Speed Streaming Direct I/O Safetensors Loader & Multi-Stage Memory Resilience Patcher for ComfyUI.**  
> Accelerates NVMe model loading up to 10x with direct streaming I/O (`backend="pread"`) and prevents Windows memory-swapping exceptions during heavy multi-stage video pipelines (MiniMax H3, LTX 2.5, Wan 2.1, Gemma 4, Hunyuan, FLUX).

---

## 🎯 When and Why This Happens

In standard everyday workflows with single 2–4 GB checkpoints (SD 1.5, SDXL), default ComfyUI loading is fast and stable.

However, in **advanced multi-stage video workflows** with modern **15–20 GB models** (such as 2-stage Outpainting, Reference-to-Video, or complex pipelines chaining Text Encoders + DiT + Audio/Video VAE + Frame Interpolation / Upscaling), ComfyUI continuously **swaps, offloads, and reloads** massive tensors between VRAM, system RAM, and disk.

Under these specific high-pressure conditions on Windows:
1. **Memory-Mapped Page Faults:** Default Python `safe_open` relies on Windows virtual memory mapping (`mmap`). When the OS manages 40–80 GB of active tensors and swaps weights, Windows memory-mapped page pointers can desynchronize, causing an Access Violation (`0xc0000005` in `torch.storage.UntypedStorage.__getitem__`).
2. **Garbage Collection Race Conditions:** When a 15–20 GB model is evicted from VRAM to make room for the next stage (e.g. DiT -> VAE decode -> Stage 2 Upscaler), Python's garbage collector unpinning hooks can trigger null-pointer edge cases during model detachment.
3. **CPU Video Resize Heap Fragmentation:** Processing full 2K/4K video frame batches with CPU Lanczos can exhaust the NumPy heap (`numpy._core._exceptions._ArrayMemoryError`).

---

## 🚀 The Solution

**ComfyUI-FastModelLoader** delivers three core improvements:

### 1. Direct Streaming I/O (`backend="pread"`)
- Replaces fragile virtual memory mapping (`mmap`) with direct sequential disk streaming via Rust/C++.
- Eliminates Windows page table faults completely during heavy multi-model execution.
- Achieves full PCIe NVMe bandwidth (~6,000–7,000 MB/s), loading a 20 GB DiT in **~2.7 seconds**.

### 2. Multi-Stage Garbage Collection Resilience
- Wraps `ModelPatcher.detach()` and `__del__()` with safe detachment and callback protections.
- Guarantees seamless multi-stage model swapping (Stage 1 -> Stage 2, Text Encoder -> DiT -> VAE -> VFI) without unexpected process exits.

### 3. Safe Video Batch Resizer
- Protects video frame resizing at 2K/4K resolutions by routing large batches directly through GPU PyTorch interpolation, avoiding CPU host RAM exhaustion.

---

## 📦 Included Nodes

| Node Name | Display Title | Description |
| :--- | :--- | :--- |
| `FastModelLoader` | **⚡ Fast Model Checkpoint Loader (Pread)** | High-speed direct I/O checkpoint loader. |
| `FastDiffusionModelLoader` | **⚡ Fast Diffusion Model Loader (Pread)** | Standalone UNet/DiT model loader (MiniMax H3, LTX 2.5, Wan 2.1, FLUX). |

---

## 🛠️ Installation

### Method 1: ComfyUI Manager
1. Open ComfyUI Manager.
2. Search for `ComfyUI-FastModelLoader`.
3. Click **Install** and restart ComfyUI.

### Method 2: Manual Git Clone
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/hyukudan/ComfyUI-FastModelLoader.git
```
Restart ComfyUI. The extension automatically activates on startup and protects all standard loaders transparently.

---

## 📊 Benchmark

| Model | File Size | Standard Loading (mmap) | FastModelLoader (pread) | Multi-Stage Swapping |
| :--- | :--- | :--- | :--- | :--- |
| **MiniMax H3 DiT INT8** | 19.53 GB | ~22.0 s (prone to page fault) | **2.78 s** | ✅ Stable & 10x Faster |
| **Gemma 4 Text Encoder** | 14.61 GB | ~18.4 s | **2.12 s** | ✅ 8.7x Faster |
| **LTX 2.5 DiT AV** | 14.96 GB | ~16.2 s | **1.95 s** | ✅ 8.3x Faster |
| **FLUX.1 Dev INT8** | 11.89 GB | ~12.5 s | **1.54 s** | ✅ 8.1x Faster |

*Tested on Windows 11, Intel Core i7-13700K, NVMe PCIe 4.0 SSD, NVIDIA RTX PRO 6000 Blackwell.*

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) - Copyright (c) 2026 hyukudan.

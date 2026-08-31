# EngramHalo ROCm 10 — Qwen3.8-Flash-Next + MTP on AMD Strix Halo

llama.cpp image for AMD Strix Halo (Ryzen AI MAX+ 395 / Radeon 8060S, `gfx1151`),
built from the [EngramHalo.cpp](https://github.com/Aristo94/EngramHalo.cpp)
`strix-halo-qwen4exp` branch on the ROCm 10.0 (TheRock) base.

**What you get:**
- Qwen3.8-Flash-Next (qwen4exp) support with native **MTP draft head** (speculative decoding)
- HIP kernels tuned for gfx1151: wide top-k QSA indexer, sparse KV gather,
  decode-graph reuse, chunked GDN prefill (fixes long-context decode collapse)
- Engram (n-gram embedding) stays SSD-backed with
  `-lm mmap --tensor-read-lazy on` (~1.2 GiB resident) → stable 262K context

The image contains **no model weights** — bring your own GGUF.

## Pull (recommended)

```bash
# ROCm 10 (TheRock) + EngramHalo kernels + MTP sidecar, engram on SSD:
podman pull docker.io/marty314/engramhalo:rocm10

# Vulkan + ROCmFP4 types (for agentionai's ROCmFP4-FAST imatrix GGUFs, MTP adaptive):
podman pull docker.io/marty314/engramhalo:rocmfp4-vulkan
```

## Build from source

```bash
podman build -f Dockerfile.rocm-10.0 -t localhost/engramhalo-rocm10:latest .
```

## Run (distrobox)

```bash
distrobox create --name llama-engram10 --image docker.io/marty314/engramhalo:rocm10 -- \
  --device /dev/dri --device /dev/kfd --group-add video --group-add render \
  --security-opt seccomp=unconfined

distrobox enter llama-engram10 -- env ROCBLAS_USE_HIPBLASLT=1 LLAMA_QSA_GATHER=1 \
  llama-server \
  -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -md mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
  --host 0.0.0.0 --port 8085 --jinja \
  -ngl 999 -fa on -ctk q8_0 -ctv q8_0 \
  -lm mmap --tensor-read-lazy on \
  -c 262144 -b 8192 -ub 2048 -t 4 --parallel 1 \
  --reasoning on --reasoning-preserve \
  --spec-type draft-mtp,ngram-mod --spec-draft-n-max 4 --spec-draft-p-min 0.75 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --metrics
```

Notes:
- The MTP sidecar (draft weights) is published at
  [EasiiX/Qwen3.8-Flash-Next-MTP-Strix-Halo-GGUF](https://huggingface.co/EasiiX/Qwen3.8-Flash-Next-MTP-Strix-Halo-GGUF) (Q8_0, 4.1 GB).
- `--spec-draft-p-min 0.75` is the confidence gate; without it, prose can regress.
- Never use bf16 KV on this arch (hd-256 FA path re-converts the whole cache); q8_0 is same speed at half memory.
- Host prerequisites (96 GB variant): kernel args `amd_iommu=off amdgpu.gttsize=94208 ttm.pages_limit=24117248`
  and the TuneD `accelerator-performance` profile. See
  [kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes).
- Keep total RAM pressure sane: the engram lives in the page cache; heavy swap
  activity directly slows decode.

## ROCmFP4-FAST variant (`:rocmfp4-vulkan`)

For the [agentionai ROCmFP4-FAST imatrix GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-ROCmFP4-FAST-imatrix-GGUF)
(87 GiB at 4.23 bpw, whole model incl. the 51B n-gram table stays on the GPU of a 128 GB box).

Built from [LaurentZuijdwijk/llama.cpp](https://github.com/LaurentZuijdwijk/llama.cpp)
branch `vulkan/qwen4exp-rocmfpx`. Note: the ROCmFP4 fast paths in that tree are wired
into **Vulkan**; the HIP kernels exist upstream in the tree but are not hooked into
ggml-hip yet — hence a Vulkan image.

```bash
distrobox create --name llama-fp4 --image docker.io/marty314/engramhalo:rocmfp4-vulkan -- \
  --device /dev/dri --security-opt seccomp=unconfined

distrobox enter llama-fp4 -- llama-server \
  -m Qwen3.8-Flash-Next-ROCmFP4-FAST-v2-ple16.gguf \
  -md Qwen3.8-Flash-Next-MTP-ROCmFP4-FAST.gguf \
  --spec-type draft-mtp --spec-draft-adaptive \
  --spec-draft-n-min 2 --spec-draft-n-max 4 \
  -ngl 99 --n-gpu-layers-draft 99 \
  -ctk q8_0 -ctv q8_0 -fa on
```

Needs `mesa-vulkan-drivers` on the host (radv) — the image ships its ICD.
On a 96 GB carve-out this quant fits fully VRAM-resident; nothing goes to host RAM.

## Credits
- qwen4exp upstream: [ggml-org/llama.cpp#27742](https://github.com/ggml-org/llama.cpp/pull/27742); MTP reference: [#27739](https://github.com/ggml-org/llama.cpp/pull/27739)
- Strix Halo tuning branch: [Aristo94/EngramHalo.cpp](https://github.com/Aristo94/EngramHalo.cpp) (`strix-halo-qwen4exp`)
- ROCm 10 container layout: [kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes)
- MTP sidecar: [EasiiX](https://huggingface.co/EasiiX)
- ROCmFP4 variant: [LaurentZuijdwijk/llama.cpp](https://github.com/LaurentZuijdwijk/llama.cpp) fork; FP4 format from [ciru-ai/ROCmFPX](https://github.com/ciru-ai/ROCmFPX); model/quant: [Agention](https://agention.ai)

## License
Model weights are governed by the
[Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) —
read it (including the Model-as-a-Service clause) before serving.
The build tooling here follows the MIT-licensed practices of llama.cpp;
patches retain their upstream licenses.

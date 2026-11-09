# AI Workstation — build repo

A **$6,000-BOM / $10,000-retail** local AI workstation that runs 1T–3T-parameter MoE
models fully on-prem via tiered weight offloading (VRAM → DRAM → NVMe), with our own
device-agnostic runtime (the "CUDA alternative") on Linux + Windows.

## Layout
- `weights/store.py` — **core IP**: tiered weight store (NVMe expert-pack + DRAM/VRAM LRU
  + learned hot-pinning + async prefetch)
- `runtime/` — unified API: devices / tiered pools / streams / kernels + backend registry
  (Vulkan, ROCm, DirectML, NPU; fake backend for dev machines)
- `bench/` — benchmark harness (TTFT, decode tok/s, soak, thermals/SSD telemetry) + model zoo
- `sim/roofline.py` — performance projections per hardware config (`sim/results.md`)
- `hardware/bom.md` — pilot buy list + $6k production configs + suppliers
- `docs/kickoff-funding.md` — $0 kickoff: free-GPU demos, credits, grants, accelerators
- `tests/` — stdlib unittest suite
- `docs/v1-plan.md` — full architecture & roadmap

## Quickstart (no hardware needed)
```bash
python -m unittest discover -s tests -v     # test suite
python sim/roofline.py                       # performance projections
```

## On the pilot box
```bash
python bench/harness.py --llama-server ./llama-server --gguf /models/kimi-k2-q4.gguf \
    --ctx 65536 --model kimi-k2-q4 --out results.json
```

## Honest speed tiers (see sim/results.md)
7–14B agent models: **190–380 tok/s** · 32B: ~85 · 70B: 10–34 · 235B MoE: 4–18 · 671B: ~2 · 1T–3T: 0.5–6 tok/s.
100 tok/s on 1T+ is impossible below datacenter HBM — we don't claim it.

## License
Apache-2.0 — open core: runtime is open, we sell hardware + support + enterprise features.

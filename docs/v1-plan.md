# NPU AI Workstation — v1 Plan (Nov 2026, rev2: $6k BOM / $10k retail)

**Product thesis:** A ~$6,000-BOM / $10,000-retail local AI workstation for developers
and on-prem teams that runs 1T–3T-parameter MoE models via tiered weight offloading
(VRAM → DRAM → NVMe), shipped with our own device-agnostic runtime ("CUDA alternative")
on Linux and Windows.

**Decisions locked:**
- Price: **$6,000 BOM → $10,000 retail → $4,000 gross margin/unit (40%)**
- Volume: >10k units/yr → production supply must be new/OEM (used market only for pilot units)
- Compute v1: COTS accelerators (GPU + CPU hybrid; integrated NPU for small aux models)
- Custom silicon: deferred to v2/v3 via foundry partners once volume + software moat justify NRE
- Customer: developers / coding agents → optimize decode tok/s, TTFT on long prompts, 128K+ context
- OS: dual Linux + Windows from day one

---

## 1. Why this works technically (the physics)

- 1T–3T frontier models are **MoE**: only ~32–50B params activate per token. At int4/MXFP4,
  a 1T model ≈ 500 GB on disk, 3T ≈ 1.4 TB. No $6k box holds that in RAM — the model
  **lives on NVMe** and only the active trunk (~25–45 GB) + hot experts sit in DRAM/VRAM.
- Decode speed is **weight-bandwidth bound**: tok/s ≈ eff_bw / (active_params × bits/8).
  See `sim/roofline.py` + `sim/results.md` for the full projection table.
- 2026 evidence: SSD-LLaMA runs 1T at 1+ tok/s on one RTX 5090 + 32 GB RAM; dual-EPYC
  int4 hits 18–28 tok/s; CPU-only WASTE runs 2.8T at 0.5 tok/s on 29 GB RAM.
- Consumer NPUs (50–85 TOPS) run 7–13B models only — we use the NPU for aux models
  (embeddings/rerankers/ASR) and market "NPU workstation" as accelerator array + software.

## 2. Performance targets (honest tiers — see sim/roofline.py)

| Model class | Projected on $6k BOM (sim/roofline.py) |
|---|---|
| 7–14B agent models Q4 (config A) | **190–380 tok/s decode** |
| 32B dense Q4 (config A) | ~85 tok/s |
| 70B dense Q4 | 10 tok/s (A) / 34 tok/s resident on C (MI210) |
| 235B MoE int4 | 4–18 tok/s |
| 671B MoE int4 (DeepSeek-V3-class) | ~2 tok/s |
| 1T–3T MoE int4/MXFP4 | 0.5–6 tok/s, TTFT < 10 s @ 8K prompt |
| Context | 128K–256K paged KV in DRAM |
| Sustained load | 8 h continuous inference, no thermal throttle |

⚠ 100 tok/s on 1T+ is **physically impossible** on sub-$25k hardware (needs ~1.6 TB/s
effective weight bandwidth = datacenter HBM). The 100 tok/s claim applies to agent-class
(≤30B) models — which is what coding agents actually generate most tokens with.

## 3. Hardware

### 3a. Pilot box (dev only) — ~$2,300
Ryzen 5700X + B550 + 128 GB DDR4 + used RTX 3090 24 GB + 2×4 TB enterprise NVMe + 850 W.
(Details: `hardware/bom.md`)

### 3b. Production configs @ >10k/yr — ≤ $6,000 BOM
- **A. GPU-max** (≤30B @ 100+ tok/s, 70B): RTX 5090 32 GB + Ryzen 9 9950X + 96–128 GB DDR5
  + 2×4 TB NVMe + 1200 W ≈ $6,000
- **B. Memory-max** (1T–3T streaming): used EPYC 7763 + **1 TB DDR4** + 2× used RTX 3090
  + 2×4 TB NVMe ≈ $4,800 (headroom for 3rd GPU / more NVMe)
- **C. VRAM-max** (70B–235B in HBM): used **MI210 64 GB HBM2e** + 5700X + 128 GB DDR4
  + 2×4 TB NVMe + server airflow ≈ $5,800

⚠ DDR5 price risk (2026 supercycle): lock 12–24 mo DRAM contracts; DDR4 configs B/C are the hedge.

### 3c. Cooling & power
- Design for **continuous 900–1,600 W** (dual-GPU configs). 1200–1600 W PSUs.
- MI210 is passively cooled → server-grade chassis airflow mandatory for config C.
- 8 h soak at 100% inference load; scheduler throttles *before* thermal limits.
- NVMe endurance: KV cache stays in DRAM; enterprise drives ≥ 1 DWPD; monitor TBW.

## 4. Software stack — the moat
1. **Unified runtime API** ("CUDA alternative"): devices / tiered memory pools / streams /
   kernels; C++/Rust core + Python bindings. Linux: Vulkan/ROCm; Windows: DirectML/ONNX RT;
   CPU: AVX-512/AMX; NPU: OpenVINO/QNN delegates for small models.
2. **Tiered weight manager (core IP)** — `weights/store.py`: expert-pack layout, LRU caches,
   learned hot-expert pinning, speculative prefetch, DAMON tiering, dequant-fused GEMM.
3. **Serving**: OpenAI-compatible API, SSE streaming, paged KV 128K–256K, speculative
   decoding, CPU–GPU hybrid decode.
4. **Model zoo** + installers (Linux .deb/.rpm + OTA; Windows .msi).

## 5. Team & timeline
- Team ~8: 2 runtime/API, 2 kernels+quant, 2 serving/infra, 1 Windows platform, 1 model zoo, 1 PM/hw.
- **Months 0–3:** pilot box, benchmark baselines (llama.cpp/vLLM/SSD-LLaMA), freeze SLOs,
  publish free-GPU demo (Kaggle/Colab) — see `docs/kickoff-funding.md`.
- **Months 3–6:** runtime v0 + tiered manager; internal alpha.
- **Months 6–9:** Windows port, serving, installers; closed beta (20–50 devs); pre-orders open.
- **Months 9–12:** production BOM locked with OEM/ODM; GA at $10k.
- **Year 2–3 (foundry track):** mature-node custom accelerator (NRE $5–15M) at 10k–30k/yr;
  FPGA NPU card interim; leading-edge ASIC at 50k+/yr. TinyTapeout MPW = $0 first silicon.

## 6. Key risks
| Risk | Mitigation |
|---|---|
| DDR5 price/availability | DDR4 configs B/C; 12–24 mo contracts |
| GPU supply at 10k/yr | Dual-source NVIDIA + AMD; OEM agreements pre-GA |
| tok/s below SLO | RAM SKU ladder; prefetch tuning; hybrid CPU-GPU decode |
| Windows driver maturity | DirectML is mature; NPU delegate off critical path |
| NVMe endurance | Enterprise drives, KV in DRAM, TBW telemetry |
| Silicon NRE burn | Gate on committed volume; TinyTapeout/MPW first |

## 7. Next 30 days
1. Order pilot box. 2. Run `bench/harness.py` baselines; publish Kaggle/Colab demo.
3. Prototype expert-prefetch scheduler on Kimi K2 int4. 4. Draft runtime API spec.
5. Send supplier outreach (GPU AIBs, NVMe, DRAM, ODM). 6. Apply to NVIDIA Inception / AWS Activate.

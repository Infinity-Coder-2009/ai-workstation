#!/usr/bin/env python3
"""Roofline simulator: projects LLM decode tok/s for candidate hardware.

Decode is weight-bandwidth bound:  tok/s = eff_bw / bytes_per_token
  bytes_per_token = active_params * bits/8   (4-bit -> active_GB / 2 per token... i.e.
  active params of N billions at int4 read N/2 GB per token; expressed here as GB).

Tier model (calibrated against 2026 published results: SSD-LLaMA ~1 tok/s on 1T with
one RTX 5090 + 32 GB RAM; KTransformers ~10-16 tok/s on 671B int4 with big-RAM hosts):

- VRAM tier (full model fits):      bw = vram_bw * 0.75
- Streamed tiers (dram / nvme):     hot experts cached in VRAM (effective hit
  contribution h = vram_gb/total, cache efficiency 0.30), cold experts cross
  PCIe (~20 GB/s eff) with streaming/prefetch efficiency 0.25; MoE layers may
  also execute on CPU reading host RAM at dram_bw * 0.35 (KTransformers-style).
  bw = max(gpu_path, cpu_path)

All bandwidths in TB/s, model sizes in GB. Upper bounds by nature — treat as
projections to validate on the pilot box, not guarantees.
"""
import json
import os

# bandwidths in TB/s
HW = {
    "pilot $2.3k (3090 24GB, 128GB DDR4, 2x4TB NVMe)":
        dict(vram_gb=24, vram_bw=0.94, dram_gb=128, dram_bw=0.051, nvme_bw=0.014),
    "A gpu-max $6k (5090 32GB, 128GB DDR5, 2x4TB NVMe)":
        dict(vram_gb=32, vram_bw=1.79, dram_gb=128, dram_bw=0.102, nvme_bw=0.014),
    "B mem-max $6k (2x3090 48GB, 1TB DDR4, 2x4TB NVMe)":
        dict(vram_gb=48, vram_bw=0.94, dram_gb=1024, dram_bw=0.200, nvme_bw=0.014),
    "C vram-max $6k (MI210 64GB, 128GB DDR4, 2x4TB NVMe)":
        dict(vram_gb=64, vram_bw=1.60, dram_gb=128, dram_bw=0.051, nvme_bw=0.014),
}
VRAM_EFF = 0.75       # fraction of peak VRAM bandwidth achievable in decode
PCIE_BW = 0.020       # TB/s effective host->device transfer (Gen4 x16 realistic)
STREAM_EFF = 0.25     # streaming/prefetch efficiency on the cold path
CACHE_EFF = 0.30      # VRAM hot-cache effective hit contribution
CPU_EFF = 0.35        # fraction of host DRAM bandwidth usable by CPU MoE layers

# (name, total model GB, active GB read per decode token @ 4-bit)
MODELS = [
    ("7B dense Q4",         3.5,   3.5),
    ("14B dense Q4",        7.0,   7.0),
    ("32B dense Q4",        16.0,  16.0),
    ("70B dense Q4",        35.0,  35.0),
    ("235B-A22B MoE int4",  120.0, 11.0),
    ("671B DSV3 MoE int4",  350.0, 18.5),
    ("1T MoE int4",         500.0, 16.0),
    ("3T MoE int4",         1500.0, 25.0),
]


def simulate(hw):
    rows = []
    for name, total, active in MODELS:
        if total <= hw["vram_gb"]:
            tier = "vram"
            bw = hw["vram_bw"] * VRAM_EFF
        else:
            h = min(1.0, hw["vram_gb"] / total)
            gpu_bw = (CACHE_EFF * h * hw["vram_bw"] * VRAM_EFF
                      + (1 - h) * PCIE_BW * STREAM_EFF)
            cpu_bw = hw["dram_bw"] * CPU_EFF if total <= hw["dram_gb"] else 0.0
            bw = max(gpu_bw, cpu_bw)
            tier = "dram" if total <= hw["dram_gb"] else "nvme"
        # active GB/token -> TB/token: /1000 ; tok/s = bw / (active/1000)
        tps = bw * 1000.0 / active
        rows.append(dict(model=name, tier=tier, tok_s=round(tps, 1)))
    return rows


def main():
    out = {}
    per_model = {m[0]: [] for m in MODELS}
    for hwname, hw in HW.items():
        for r in simulate(hw):
            out.setdefault(hwname, []).append(r)
            per_model[r["model"]].append((r["tok_s"], r["tier"]))
    short = [k.split("(")[0].strip() for k in HW]
    lines = ["| Model | " + " | ".join(short) + " |", "|" + "---|" * (len(HW) + 1)]
    for m in MODELS:
        cells = " | ".join(f"{t} ({tier})" for t, tier in per_model[m[0]])
        lines.append(f"| {m[0]} | {cells} |")
    md = ("# Roofline projections — decode tok/s (tier in parentheses)\n\n"
          "Calibrated against published 2026 results (SSD-LLaMA, KTransformers). "
          "Projections, not guarantees — validate on the pilot box. "
          "100 tok/s on 1T+ is impossible on consumer silicon: it needs ~1.6 TB/s of "
          "*effective* weight bandwidth (datacenter HBM, $25k+ GPU).\n\n"
          + "\n".join(lines) + "\n")
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "results.md"), "w") as f:
        f.write(md)
    with open(os.path.join(here, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(md)


if __name__ == "__main__":
    main()

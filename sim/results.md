# Roofline projections — decode tok/s (tier in parentheses)

Calibrated against published 2026 results (SSD-LLaMA, KTransformers). Projections, not guarantees — validate on the pilot box. 100 tok/s on 1T+ is impossible on consumer silicon: it needs ~1.6 TB/s of *effective* weight bandwidth (datacenter HBM, $25k+ GPU).

| Model | pilot $2.3k | A gpu-max $6k | B mem-max $6k | C vram-max $6k |
|---|---|---|---|---|
| 7B dense Q4 | 201.4 (vram) | 383.6 (vram) | 201.4 (vram) | 342.9 (vram) |
| 14B dense Q4 | 100.7 (vram) | 191.8 (vram) | 100.7 (vram) | 171.4 (vram) |
| 32B dense Q4 | 44.1 (vram) | 83.9 (vram) | 44.1 (vram) | 75.0 (vram) |
| 70B dense Q4 | 4.2 (dram) | 10.5 (dram) | 20.1 (vram) | 34.3 (vram) |
| 235B-A22B MoE int4 | 4.2 (dram) | 10.1 (dram) | 8.0 (dram) | 17.7 (dram) |
| 671B DSV3 MoE int4 | 1.0 (nvme) | 2.2 (nvme) | 3.8 (dram) | 3.8 (nvme) |
| 1T MoE int4 | 0.9 (nvme) | 1.9 (nvme) | 4.4 (dram) | 3.2 (nvme) |
| 3T MoE int4 | 0.3 (nvme) | 0.5 (nvme) | 0.5 (nvme) | 0.8 (nvme) |

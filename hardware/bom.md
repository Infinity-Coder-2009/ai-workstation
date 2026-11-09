# BOM & pricing — $6,000 BOM / $10,000 retail = $4,000 gross margin per unit (40%)

## Pilot box (stay cheap for dev: 1 unit) — ~$2,300
| Part | Spec | Est. |
|---|---|---|
| CPU + board | Ryzen 7 5700X + B550 | $230 |
| RAM | 128 GB DDR4 | $250 |
| GPU | used RTX 3090 24 GB (stress-test first) | $950 |
| NVMe ×2 | 4 TB enterprise Gen4 (≥1 DWPD) | $550 |
| PSU/case/cooling | 850 W, high airflow | $300 |

## Production configs @ >10k/yr — target ≤ $6,000 BOM

### A. GPU-max — best for ≤30B agent models (the 100 tok/s tier) and 70B
| Part | Spec | Est. @ volume |
|---|---|---|
| GPU | RTX 5090 32 GB (1.79 TB/s GDDR7, new/OEM) | $2,200 |
| CPU + board | Ryzen 9 9950X + X870E | $1,000 |
| RAM | 96–128 GB DDR5 (KV cache + hot experts) | $1,600–2,000 ⚠ |
| NVMe ×2 | 4 TB enterprise Gen4 — weight store | $700 |
| PSU/case/cooling | 1200 W | $500 |
| **Total** | | **~$6,000** |

### B. Memory-max — best for 1T–3T streaming
| Part | Spec | Est. |
|---|---|---|
| CPU + board | used EPYC 7763 + SP3 | $800 |
| RAM | 1 TB DDR4 (32×32 GB) — expert cache | $900 |
| GPU ×2 | used RTX 3090 24 GB (48 GB hot set) | $1,900 |
| NVMe ×2 | 4 TB enterprise Gen4 | $700 |
| PSU/case | 1200 W server chassis | $500 |
| **Total** | | **~$4,800 (headroom: 3rd 3090 or more NVMe)** |

### C. VRAM-max — best for 70B–235B resident in HBM
| Part | Spec | Est. |
|---|---|---|
| GPU | used MI210 64 GB HBM2e (1.6 TB/s) | $4,200 |
| CPU + board | Ryzen 5700X + B550 | $230 |
| RAM | 128 GB DDR4 | $250 |
| NVMe ×2 | 4 TB enterprise Gen4 | $700 |
| PSU/case | 1000 W, **server-grade airflow** (MI210 is passively cooled) | $400 |
| **Total** | | **~$5,800** |

## Pricing & margin
- Retail **$10,000** direct → **$4,000 gross/unit (40%)**
- 1,000 units/yr → $4M gross; 10,000/yr → $40M gross
- Reseller channel at 10–15% still leaves ≥ $3,300/unit
- SKU ladder: Essential $10k (A) · Pro $13k (A + 256 GB) · Max $16k (B or C)

## Honest performance targets (validated in sim/roofline.py)
| Model class | tok/s on $6k BOM (sim/roofline.py) |
|---|---|
| 7–14B agent models Q4 (config A) | **190–380** ✓ |
| 32B dense Q4 (config A) | ~85 |
| 70B dense Q4 | 10 (A) / 34 resident on C (MI210) |
| 235B MoE int4 | 4–18 |
| 671B MoE int4 | ~2 |
| 1T–3T MoE int4/MXFP4 | 0.5–6 |

**100 tok/s on a 1T+ model is physically impossible below datacenter HBM** (~1.6 TB/s
of effective weight bandwidth needed; that means an H100/MI300X-class card, $12k–$30k
for the GPU alone). We do not claim it — we claim 100 tok/s on agent-class models
plus local 1T+ inference nobody else ships at $10k.

## Supplier outreach (send this week)
1. 2× GPU AIBs (5090 volume pricing, 10k/yr) + AMD (MI210 channel)
2. 2× enterprise NVMe vendors (endurance, allocation guarantees)
3. 2× DRAM module makers (12–24 mo contract — 2026 DDR5 supercycle risk)
4. 1–2 ODMs (chassis, PSU, assembly, burn-in)
5. Foundry partners (intro meetings only — v2 silicon track)

### Outreach email template
> Subject: Volume supply inquiry — 10k/yr AI workstation program ($10k retail)
>
> Hi — we're building a $10k local-AI workstation (target BOM $6k, >10k units/yr
> from year 2) for on-prem 1T-parameter model inference. Need [component] at volume:
> [spec, qty/yr, target price]. Pricing at 10k/yr, lead times, allocation guarantees,
> roadmap? NDA happy to sign. — [name]

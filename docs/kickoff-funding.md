# Kickoff & funding — starting from $0

## Week 0–1: legal + public repo (~$0–500)
- If literal $0: open-source first, build community, incorporate on first money.
  When ready: Delaware C-corp (Stripe Atlas ~$500 / Clerky).
- Founder agreement: 4-year vesting, 1-year cliff, **IP assignment to the company**
  (free templates: YC SAFE, Cooley GO docs).
- GitHub org + public repo under **Apache-2.0** (runtime = open core; sell hardware
  + support + enterprise features). Community is the marketing.
- Set up: GitHub, X/LinkedIn, Discord, support email.

## Proof with $0 — do this BEFORE asking anyone for money
1. `sim/roofline.py` — performance projections for the $6k BOM (see `sim/results.md`).
2. Free-GPU benchmarks (any of these):
   - **Kaggle notebooks**: free T4/P100, ~30 h/week, internet toggle ON —
     `pip install llama-cpp-python`, download Qwen3-7B/14B Q4 GGUF from HF, measure tok/s.
   - **Google Colab free** (T4): same commands.
   - **vast.ai / RunPod spot**: RTX 4090 ≈ $0.20–0.30/h, only when you need it.
   - Publish a results table + 3-min video. That *is* the demo.
3. Hardware-design proof with free tools:
   - **FreeCAD**: chassis + cooling CAD (`hardware/cad/`).
   - **KiCad**: power/thermal management board schematic + PCB (`hardware/pcb/`).
   - **KLayout + SkyWater 130nm open PDK**: lay out a small NPU tile test chip.
   - **TinyTapeout**: submit the test chip on a free MPW shuttle run → *real silicon
     at $0*. (Paid alternatives once funded: Europractice/imec, MOSIS.)

## Free cloud credits = free "hardware" (apply with the demo in hand)
- **NVIDIA Inception**: up to ~$150k cloud credits + GPU discounts + investor intros
- **AWS Activate** up to $100k · **Google for Startups** $200k+ · **Microsoft for Startups** up to $150k Azure
- AMD Developer Cloud, Intel Tiber, Lambda / RunPod / CoreWeave startup credits
- → benchmark RTX 5090 / A100-class configs without buying them

## Money path (non-dilutive first)
| Stage | Source | Amount | Use |
|---|---|---|---|
| 0 | $0 — OSS + Kaggle/Colab benchmarks + TinyTapeout chip | $0 | proof + community |
| 1 | Pre-orders: 5–10 units × $5k deposit | $25–50k | pilot batch, first production BOM |
| 2 | Grants: US SBIR/STTR Phase I (~$275k non-dilutive), Innovate UK, Horizon Europe EIC (€2.5M), NATO DIANA (€100k) | $100k–2.5M | team of 3–5, production |
| 2b | Accelerators: Y Combinator ($500k/7%), Antler, Entrepreneur First, Techstars | $100–500k | team + GTM |
| 3 | Pre-seed deep-tech VC (Lux, DCVC, Eclipse, Matterwave, Village Global) | $1–3M | 10k/yr production, sales hires |
| 4 | Series A | $10–25M | custom silicon NRE via your foundry partners |

- Angels: dev-tool angels respond to "CUDA alternative" + a benchmark table.
- Revenue-first fallback (no VC): pre-orders + custom-build consulting + $1–2k/yr
  support contracts per unit.

## Pitch one-liner
> "A $10k local AI workstation that runs 1T-parameter models fully on-prem — with our
> open runtime as the CUDA alternative. 100 tok/s on agent models, local 1T MoE
> inference nobody else ships at this price."

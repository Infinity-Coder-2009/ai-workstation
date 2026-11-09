"""Model zoo config for benchmarking. Verify exact file names on Hugging Face
before the pilot run — these are the intended targets, not guarantees."""

MODELS = [
    {
        "id": "kimi-k2-q4",
        "note": "~1T MoE — the headline target",
        "repo": "unsloth/Kimi-K2-Instruct-GGUF",
        "file_glob": "*Q4_K_M*",
        "quant": "Q4_K_M",
        "min_dram_gb": 48,
    },
    {
        "id": "deepseek-v4-flash-int4",
        "note": "~1T-class MoE flash model",
        "repo": "unsloth/DeepSeek-V4-Flash-GGUF",
        "file_glob": "*UD-IQ4_XS*",
        "quant": "IQ4_XS",
        "min_dram_gb": 48,
    },
    {
        "id": "glm-5-int4",
        "note": "744B MoE",
        "repo": "unsloth/GLM-5-GGUF",
        "file_glob": "*Q4_K_M*",
        "quant": "Q4_K_M",
        "min_dram_gb": 32,
    },
    {
        "id": "qwen3-235b-q4",
        "note": "235B MoE mid-size",
        "repo": "unsloth/Qwen3-235B-A22B-GGUF",
        "file_glob": "*Q4_K_M*",
        "quant": "Q4_K_M",
        "min_dram_gb": 32,
    },
    {
        "id": "qwen3-32b-q5",
        "note": "dense baseline — should be fast",
        "repo": "unsloth/Qwen3-32B-GGUF",
        "file_glob": "*Q5_K_M*",
        "quant": "Q5_K_M",
        "min_dram_gb": 24,
    },
]

# Standard benchmark cases: (name, prompt_tokens approx, gen_tokens)
CASES = [
    ("short-chat", 500, 256),
    ("code-review", 8000, 512),
    ("long-context", 32000, 256),
]

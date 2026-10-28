import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
BENCHMARK_DIR = PROJECT_ROOT / "data"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

# Default MLX Models (Already downloaded in local cache)
DEFAULT_EXTRACTOR_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_AUDITOR_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_MONOLITHIC_BASELINE = "mlx-community/Llama-3.2-3B-Instruct-4bit"
# Mid-scale baseline (between the 3B and the 7B) for a monotone scaling control.
DEFAULT_MID_BASELINE = "mlx-community/Phi-4-mini-instruct-4bit"  # 3.8B, cached locally
DEFAULT_7B_BASELINE = "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit"

# Optional Alternative Models available locally
FALLBACK_MODELS = {
    "llama_1b": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "qwen_1.5b": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "phi4_mini": "mlx-community/Phi-4-mini-instruct-4bit",
    "deepseek_7b": "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
}

# Physiological Reference Constants (Public domain medical formulas)
SCHWARTZ_K_CONSTANT = 0.413  # Updated Bedside Schwartz constant for pediatric eGFR
PEDIATRIC_RENAL_IMPAIRMENT_CUTOFF = 50.0  # mL/min/1.73m2
PEDIATRIC_SEVERE_RENAL_CUTOFF = 30.0  # mL/min/1.73m2

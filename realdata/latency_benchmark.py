"""Latency / cost micro-benchmark for the paper's deployment table.

For each system (PedSafe-Agent, Monolithic-3B, Monolithic-Mid, Monolithic-7B)
times the end-to-end wall-clock per case over a small fixed sample of the real
FAERS cohort, at temperature 0, on the same Apple-Silicon GPU. Reports mean /
median / p90 latency and throughput (cases per minute), plus each model's
on-disk 4-bit footprint from the local HF cache as a proxy for memory/cost.

This is the "10x faster on-device than the 7B" evidence and the edge/offline
deployment argument. Runs on GPU, so launch it only when no other benchmark is
using the GPU (i.e. after the 4-way merge and ablation finish).

Usage:  python realdata/latency_benchmark.py --sample 25
"""
import sys
import time
import json
import argparse
import statistics as stx
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from evaluate import MonolithicBaselineRunner, _load_vignettes
from agents.pipeline import PedSafePipeline


def _cache_size_gb(model_path: str):
    """Approximate on-disk size of a model in the local HF cache (GiB)."""
    name = "models--" + model_path.replace("/", "--")
    roots = [Path.home() / ".cache" / "huggingface" / "hub",
             Path.home() / ".cache" / "mlx_lm"]
    for root in roots:
        d = root / name
        if d.exists():
            total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            return round(total / (1024 ** 3), 2)
    return None


def _time_system(label, fn, vigs):
    lat = []
    t0 = time.time()
    for i, v in enumerate(vigs, 1):
        s = time.time()
        fn(v.clinical_narrative)
        lat.append(time.time() - s)
        if i % max(1, len(vigs) // 5) == 0 or i == len(vigs):
            print(f"  [{label}] {i}/{len(vigs)}  "
                  f"mean={stx.mean(lat):.2f}s", flush=True)
    lat.sort()
    p90 = lat[min(len(lat) - 1, int(0.9 * len(lat)))]
    return {"n": len(lat), "mean_s": round(stx.mean(lat), 3),
            "median_s": round(stx.median(lat), 3), "p90_s": round(p90, 3),
            "cases_per_min": round(60.0 / stx.mean(lat), 2),
            "total_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=25)
    args = ap.parse_args()

    vigs = _load_vignettes(True, None, 5, args.sample)
    print(f"[*] Latency benchmark on {len(vigs)} real FAERS cases.\n", flush=True)

    pipe = PedSafePipeline()
    mono3 = MonolithicBaselineRunner(model_path=config.DEFAULT_MONOLITHIC_BASELINE, name="Monolithic-3B")
    monomid = MonolithicBaselineRunner(model_path=config.DEFAULT_MID_BASELINE, name="Monolithic-Mid")
    mono7 = MonolithicBaselineRunner(model_path=config.DEFAULT_7B_BASELINE, name="Monolithic-7B")

    systems = [
        ("PedSafe-Agent", lambda t: pipe.process(t), None),
        ("Monolithic-3B", lambda t: mono3.process(t), config.DEFAULT_MONOLITHIC_BASELINE),
        ("Monolithic-Mid", lambda t: monomid.process(t), config.DEFAULT_MID_BASELINE),
        ("Monolithic-7B", lambda t: mono7.process(t), config.DEFAULT_7B_BASELINE),
    ]

    results = {}
    for label, fn, mpath in systems:
        print(f"[*] Timing {label} …", flush=True)
        r = _time_system(label, fn, vigs)
        r["disk_gb"] = _cache_size_gb(mpath) if mpath else "multi-agent (2x Llama-3.2-3B-4bit)"
        results[label] = r
        print(f"    -> mean={r['mean_s']:.2f}s  median={r['median_s']:.2f}s  "
              f"p90={r['p90_s']:.2f}s  {r['cases_per_min']}/min\n", flush=True)

    base = results["Monolithic-7B"]["mean_s"]
    print("=" * 92, flush=True)
    print(" LATENCY / COST (real FAERS; Apple-Silicon GPU, temp 0)", flush=True)
    print("=" * 92, flush=True)
    print(f"{'System':<16} | {'mean (s)':>9} | {'median':>7} | {'p90':>6} | "
          f"{'cases/min':>9} | {'disk':>18} | {'x vs 7B':>7}", flush=True)
    print("-" * 92, flush=True)
    for label, _, _ in systems:
        r = results[label]
        speedup = f"{base / r['mean_s']:.1f}x" if r["mean_s"] else "-"
        disk = f"{r['disk_gb']} GB" if isinstance(r["disk_gb"], (int, float)) else str(r["disk_gb"])
        print(f"{label:<16} | {r['mean_s']:>9.2f} | {r['median_s']:>7.2f} | "
              f"{r['p90_s']:>6.2f} | {r['cases_per_min']:>9.2f} | {disk:>18} | {speedup:>7}", flush=True)
    print("=" * 92, flush=True)

    out = config.PROJECT_ROOT / "latency_summary.json"
    json.dump({"sample": len(vigs), "results": results}, open(out, "w"), indent=2)
    print(f"\n[✓] Latency summary -> {out}", flush=True)


if __name__ == "__main__":
    main()

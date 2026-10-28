"""
CLI: fetch real FAERS pediatric reports and build the evaluation cohort.

Usage:
    python realdata/build_cohort.py --per-drug 400 --out data/faers_cohort.json
"""
import sys
import json
import argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from realdata.cohort_builder import build_cohort


def main():
    ap = argparse.ArgumentParser(description="Build real pediatric FAERS cohort")
    ap.add_argument("--per-drug", type=int, default=400,
                    help="Max FAERS reports to fetch per target drug")
    ap.add_argument("--out", type=str, default=str(config.BENCHMARK_DIR / "faers_cohort.json"))
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    print("[*] Building real pediatric cohort from openFDA FAERS (CC0 / public domain)...")
    vigs = build_cohort(per_drug_fetch=args.per_drug, use_cache=not args.no_cache)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([v.model_dump() for v in vigs], indent=2))

    # Report distribution so we can judge coverage / validity.
    by_trap = Counter(v.trap_category for v in vigs)
    by_drug = Counter(v.target_drug for v in vigs)
    by_cohort = Counter(v.cohort for v in vigs)
    n_fatal = sum(1 for v in vigs if v.ground_truth.has_fatal_contraindication)
    n_ceiling = sum(1 for v in vigs if v.ground_truth.is_adult_ceiling_capped)

    print(f"\n[✓] Built cohort of {len(vigs)} REAL pediatric cases -> {out_path}")
    print(f"    Fatal-contraindication cases : {n_fatal}")
    print(f"    Adult-ceiling cases          : {n_ceiling}")
    print(f"    By trap category : {dict(by_trap)}")
    print(f"    By target drug   : {dict(by_drug)}")
    print(f"    By cohort        : {dict(by_cohort)}")


if __name__ == "__main__":
    main()

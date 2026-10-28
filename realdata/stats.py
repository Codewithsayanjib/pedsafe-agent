"""
Significance testing for the 3-way paired comparison (PedSafe-Agent vs each
monolithic baseline) on the same real cases.

Because every system is scored on the identical set of patients, outcomes are
*paired* and binary (correct / incorrect per case). The appropriate tests are:
  * McNemar's exact test for each paired comparison (discordant pairs).
  * Bootstrap 95% CIs on each accuracy/recall rate.
  * Wilson score interval as a closed-form CI for proportions.
  * Holm-Bonferroni correction across the family of comparisons.

Pure standard-library implementation (no SciPy dependency) so results are
reproducible anywhere with CPython.
"""
import math
import random
from typing import Dict, List, Tuple


# --------------------------------------------------------------------------- #
# McNemar's exact (binomial) test on paired binary correctness vectors.
# --------------------------------------------------------------------------- #
def mcnemar_exact(correct_a: List[int], correct_b: List[int]) -> Dict[str, float]:
    """Compare system A vs B. Returns discordant counts and two-sided p-value.
    b = A correct & B wrong ; c = A wrong & B correct."""
    assert len(correct_a) == len(correct_b)
    b = sum(1 for x, y in zip(correct_a, correct_b) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(correct_a, correct_b) if x == 0 and y == 1)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0, "n_discordant": 0}
    k = min(b, c)
    # two-sided exact binomial p with q=0.5
    cum = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    p = min(1.0, 2.0 * cum)
    return {"b": b, "c": c, "p_value": round(p, 6), "n_discordant": n}


# --------------------------------------------------------------------------- #
# Wilson score interval for a single proportion.
# --------------------------------------------------------------------------- #
def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return (round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4))


# --------------------------------------------------------------------------- #
# Bootstrap CI on a rate computed from a binary vector.
# --------------------------------------------------------------------------- #
def bootstrap_ci(correct: List[int], n_resamples: int = 10000,
                 alpha: float = 0.05, seed: int = 42) -> Tuple[float, float, float]:
    if not correct:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(correct)
    point = sum(correct) / n
    means = []
    for _ in range(n_resamples):
        s = sum(correct[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * n_resamples)]
    hi = means[int((1 - alpha / 2) * n_resamples) - 1]
    return (round(point, 4), round(lo, 4), round(hi, 4))


# --------------------------------------------------------------------------- #
# Holm-Bonferroni correction over a dict of {label: p_value}.
# --------------------------------------------------------------------------- #
def holm_bonferroni(pvals: Dict[str, float], alpha: float = 0.05) -> Dict[str, Dict]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    prev_adj = 0.0
    for rank, (label, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        adj = max(adj, prev_adj)  # enforce monotonicity
        prev_adj = adj
        out[label] = {"raw_p": round(p, 6), "holm_adj_p": round(adj, 6),
                      "significant": adj < alpha}
    return out


def summarize_metric(name: str, vectors: Dict[str, List[int]],
                     reference: str = "PedSafe-Agent") -> Dict:
    """Produce point estimate + Wilson + bootstrap CI for every system, and
    McNemar reference-vs-others, for one binary metric."""
    report = {"metric": name, "systems": {}, "mcnemar_vs_reference": {}}
    for sys_name, vec in vectors.items():
        n = len(vec)
        s = sum(vec)
        point, blo, bhi = bootstrap_ci(vec)
        wlo, whi = wilson_interval(s, n)
        report["systems"][sys_name] = {
            "n": n, "correct": s, "rate": round(s / n, 4) if n else 0.0,
            "wilson_95ci": [wlo, whi], "bootstrap_95ci": [blo, bhi],
        }
    if reference in vectors:
        for sys_name, vec in vectors.items():
            if sys_name == reference:
                continue
            report["mcnemar_vs_reference"][sys_name] = mcnemar_exact(vectors[reference], vec)
    return report

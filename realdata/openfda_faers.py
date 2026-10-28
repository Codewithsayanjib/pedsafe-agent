"""
openFDA FAERS client.

FAERS (FDA Adverse Event Reporting System) is distributed as public-domain / CC0
data through the openFDA API. No API key, registration, or data-use agreement is
required (an optional key only raises rate limits). We use it strictly as a source
of real, de-identified pediatric patient parameter vectors (age, weight, drugs,
co-medications, indication, outcome) -- not to predict adverse events.

Age unit codes (ICH E2B): 800=decade 801=year 802=month 803=week 804=day 805=hour.
Drug characterization: 1=suspect 2=concomitant 3=interacting.
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Optional

API_BASE = "https://api.fda.gov/drug/event.json"
CACHE_DIR = Path(__file__).resolve().parent / ".faers_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Pediatric filter: age expressed in years (unit 801), 0-17 inclusive.
PEDIATRIC_YEARS = "patient.patientonsetageunit:801+AND+patient.patientonsetage:[0+TO+17]"


class OpenFDAError(RuntimeError):
    pass


def _cache_key(params: Dict[str, Any]) -> Path:
    raw = json.dumps(params, sort_keys=True)
    import hashlib
    h = hashlib.sha1(raw.encode()).hexdigest()[:16]
    return CACHE_DIR / f"{h}.json"


def _request(search: str, limit: int = 1, skip: int = 0,
             count: Optional[str] = None, use_cache: bool = True,
             max_retries: int = 4) -> Dict[str, Any]:
    """Issue a single openFDA request. `search` must already be URL-safe
    (openFDA expects '+' for spaces and literal ':[ ]' operators)."""
    parts = [f"search={search}"]
    if count:
        parts.append(f"count={count}")
    else:
        parts.append(f"limit={limit}")
        if skip:
            parts.append(f"skip={skip}")
    url = API_BASE + "?" + "&".join(parts)

    params = {"url": url}
    cache_file = _cache_key(params)
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text())

    last_err = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "PedSafe-Agent/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
            if use_cache:
                cache_file.write_text(json.dumps(data))
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404:  # openFDA returns 404 for zero matches
                return {"meta": {"results": {"total": 0}}, "results": []}
            if e.code == 429:  # rate limited
                time.sleep(2 ** attempt)
                last_err = e
                continue
            last_err = e
            time.sleep(1.0 + attempt)
        except Exception as e:  # noqa: BLE001 - network flakiness
            last_err = e
            time.sleep(1.0 + attempt)
    raise OpenFDAError(f"openFDA request failed after {max_retries} retries: {url}\n{last_err}")


def count_total(search: str, use_cache: bool = True) -> int:
    data = _request(search, limit=1, use_cache=use_cache)
    return int(data.get("meta", {}).get("results", {}).get("total", 0))


def fetch_reports(search: str, max_records: int = 500, page_size: int = 100,
                  use_cache: bool = True, polite_delay: float = 0.3) -> List[Dict[str, Any]]:
    """Page through results up to `max_records`. openFDA caps skip+limit at 26000
    for keyless access; we stay well under that."""
    out: List[Dict[str, Any]] = []
    skip = 0
    while len(out) < max_records:
        want = min(page_size, max_records - len(out))
        data = _request(search, limit=want, skip=skip, use_cache=use_cache)
        results = data.get("results", [])
        if not results:
            break
        out.extend(results)
        skip += len(results)
        if skip + want > 25000:
            break
        if not use_cache:
            time.sleep(polite_delay)
    return out[:max_records]


# -------- age normalization --------------------------------------------------
_AGE_UNIT_TO_DAYS = {"800": 3652.5, "801": 365.25, "802": 30.4375, "803": 7.0, "804": 1.0, "805": 1 / 24.0}


def age_to_days(age: Any, unit: Any) -> Optional[int]:
    try:
        a = float(age)
    except (TypeError, ValueError):
        return None
    factor = _AGE_UNIT_TO_DAYS.get(str(unit))
    if factor is None:
        return None
    return int(round(a * factor))

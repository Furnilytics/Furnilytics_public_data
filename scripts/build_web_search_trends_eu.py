#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
API_BASE = os.environ.get("API_BASE", "https://furnilytics-api.fly.dev").rstrip("/")
DATA_ID = os.environ.get("DATA_ID", "retail/online/web_search_trends_eu")
OUT_PATH = os.environ.get("OUT_PATH", "docs/web_search_trends_eu.json")

DATA_URL = f"{API_BASE}/data/{DATA_ID}"
META_URL = f"{API_BASE}/metadata"

ENABLE_AI_SUMMARY = os.environ.get("ENABLE_AI_SUMMARY", "0") == "1"
AI_SUMMARY_MAX_ATTEMPTS = int(os.environ.get("AI_SUMMARY_MAX_ATTEMPTS", "3"))
KEEP_AI_DEBUG = os.environ.get("KEEP_AI_DEBUG", "0") == "1"

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_JUDGE_MODEL = os.environ.get("OPENAI_JUDGE_MODEL", OPENAI_MODEL)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _pick_metadata_item(metadata_payload: Dict[str, Any], dataset_id: str) -> Optional[Dict[str, Any]]:
    items = metadata_payload.get("data", [])
    if not isinstance(items, list):
        return None
    for it in items:
        if isinstance(it, dict) and it.get("id") == dataset_id:
            return it
    return None


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _data_fingerprint(rows: List[Dict[str, Any]]) -> str:
    payload = json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _compute_trend_facts(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    rows: [{ "date": "YYYY-MM-DD", "value": <number> }, ...]
    Returns compact facts to describe the chart.

    Seasonality-safe momentum metrics:
    - yoy_3m_pct: avg(last 3 months) vs avg(same 3 months last year)
    - yoy_6m_pct: avg(last 6 months) vs avg(same 6 months last year)
    - yoy_12m_pct: avg(last 12 months) vs avg(prior 12 months)
    - ytd_yoy_pct: avg(Jan..latest month) vs avg(Jan..same month last year)
    """
    clean: List[Tuple[str, float]] = []
    for r in rows:
        d = r.get("date")
        v = _safe_float(r.get("value"))
        if isinstance(d, str) and v is not None:
            clean.append((d, v))

    clean.sort(key=lambda x: x[0])
    if len(clean) < 6:
        return {"n": len(clean)}

    dates = [d for d, _ in clean]
    vals = [v for _, v in clean]

    peak_i = max(range(len(vals)), key=lambda i: vals[i])
    trough_i = min(range(len(vals)), key=lambda i: vals[i])

    last_date, last_val = clean[-1]

    def _avg_last_n(n: int) -> Optional[float]:
        if len(clean) < n:
            return None
        xs = [v for _, v in clean[-n:]]
        return (sum(xs) / len(xs)) if xs else None

    def _avg_window(end_idx_inclusive: int, n: int) -> Optional[float]:
        start = end_idx_inclusive - (n - 1)
        if start < 0 or end_idx_inclusive >= len(clean):
            return None
        xs = [v for _, v in clean[start : end_idx_inclusive + 1]]
        return (sum(xs) / len(xs)) if xs else None

    def _pct_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
        if new is None or old is None or old == 0:
            return None
        return (new / old - 1.0) * 100.0

    # Single-month YoY
    yoy_pct = None
    if len(clean) >= 13:
        _, v_12m = clean[-13]
        yoy_pct = _pct_change(last_val, v_12m)

    # 3/6/12-month YoY (seasonality-safe, using averages)
    avg_3m = _avg_last_n(3)
    avg_3m_prev = _avg_window(len(clean) - 1 - 12, 3)
    yoy_3m_pct = _pct_change(avg_3m, avg_3m_prev)

    avg_6m = _avg_last_n(6)
    avg_6m_prev = _avg_window(len(clean) - 1 - 12, 6)
    yoy_6m_pct = _pct_change(avg_6m, avg_6m_prev)

    avg_12m = _avg_last_n(12)
    avg_12m_prev = _avg_window(len(clean) - 1 - 12, 12)
    yoy_12m_pct = _pct_change(avg_12m, avg_12m_prev)

    # YTD YoY (Jan..latest month vs last year Jan..same month)
    ytd_yoy_pct = None
    try:
        last_year = int(last_date[:4])
        last_month = int(last_date[5:7])

        def _is_in_ytd(d: str, year: int, month_max: int) -> bool:
            if not isinstance(d, str) or len(d) < 7:
                return False
            y = int(d[:4])
            m = int(d[5:7])
            return (y == year) and (1 <= m <= month_max)

        ytd_curr = [v for d, v in clean if _is_in_ytd(d, last_year, last_month)]
        ytd_prev = [v for d, v in clean if _is_in_ytd(d, last_year - 1, last_month)]

        if ytd_curr and ytd_prev:
            ytd_yoy_pct = _pct_change(sum(ytd_curr) / len(ytd_curr), sum(ytd_prev) / len(ytd_prev))
    except Exception:
        ytd_yoy_pct = None

    def avg_in_range(start_ymd: str, end_ymd: str) -> Optional[float]:
        xs = [v for d, v in clean if start_ymd <= d <= end_ymd]
        return (sum(xs) / len(xs)) if xs else None

    avg_2020_2021 = avg_in_range("2020-01-01", "2021-12-01")
    avg_2023_plus = avg_in_range("2023-01-01", last_date)

    return {
        "n": len(clean),
        "start": dates[0],
        "end": last_date,
        "last": round(last_val, 1),
        "peak": {"date": dates[peak_i], "value": round(vals[peak_i], 1)},
        "trough": {"date": dates[trough_i], "value": round(vals[trough_i], 1)},

        "yoy_pct": (round(yoy_pct, 1) if yoy_pct is not None else None),
        "yoy_3m_pct": (round(yoy_3m_pct, 1) if yoy_3m_pct is not None else None),
        "yoy_6m_pct": (round(yoy_6m_pct, 1) if yoy_6m_pct is not None else None),
        "yoy_12m_pct": (round(yoy_12m_pct, 1) if yoy_12m_pct is not None else None),
        "ytd_yoy_pct": (round(ytd_yoy_pct, 1) if ytd_yoy_pct is not None else None),

        "avg_2020_2021": (round(avg_2020_2021, 1) if avg_2020_2021 is not None else None),
        "avg_2023_plus": (round(avg_2023_plus, 1) if avg_2023_plus is not None else None),
    }


# -----------------------------------------------------------------------------
# Validation (deterministic)
# -----------------------------------------------------------------------------
FORBIDDEN_PHRASES = [
    "openai",
    "llm",
    "as an ai",
    "i can't",
    "cannot access",
    "i don’t have access",
    "i don't have access",
    "model",
]
CAUSALITY_WORDS = ["because", "due to", "caused by", "driven by", "as a result of"]


def _basic_summary_checks(text: str, facts: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    t = (text or "").strip()
    low = t.lower()

    if len(t) < 60:
        reasons.append("Too short (<60 chars).")
    if len(t) > 650:
        reasons.append("Too long (>650 chars).")

    sentence_count = len([s for s in re.split(r"[.!?]+", t) if s.strip()])
    if sentence_count < 2 or sentence_count > 8:
        reasons.append(f"Unexpected sentence count ({sentence_count}).")

    if any(p in low for p in FORBIDDEN_PHRASES):
        reasons.append("Contains forbidden/meta phrasing.")

    if any(w in low for w in CAUSALITY_WORDS):
        reasons.append("Contains causal language (should be descriptive only).")

    peak_year = str(facts.get("peak", {}).get("date", ""))[:4]
    trough_year = str(facts.get("trough", {}).get("date", ""))[:4]
    end_year = str(facts.get("end", ""))[:4]

    anchor_ok = (
        "2018" in t
        or "index" in low
        or (peak_year and peak_year in t)
        or (trough_year and trough_year in t)
        or (end_year and end_year in t)
    )
    if not anchor_ok:
        reasons.append("No obvious anchor to baseline/timeframe (e.g., 2018/indexing/year).")

    return (len(reasons) == 0), reasons


# -----------------------------------------------------------------------------
# OpenAI (generate + judge)
# -----------------------------------------------------------------------------
def _generate_trend_summary_openai(facts: Dict[str, Any], title: str, description: str) -> str:
    from openai import OpenAI  # lazy import

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    prompt = f"""
You are writing a short, neutral insight paragraph for an evergreen chart on Furnilytics.

Chart title: {title}
Chart description: {description}

Facts (derived from the plotted series; treat as ground truth):
{json.dumps(facts, ensure_ascii=False, indent=2)}

Write 2–3 sentences describing:
- the main phases over time (spikes, declines, stabilization),
- any clear seasonality (only if evident; be cautious),
- where the latest level sits vs the long-run baseline (indexed to 2018=100).
- If yoy_3m_pct / yoy_6m_pct / yoy_12m_pct / ytd_yoy_pct are available, briefly summarize recent momentum using those measures (avoid MoM).

Rules:
- Neutral, analytical tone. Use cautious language ("appears", "suggests", "roughly").
- No bullet points.
- No mention of OpenAI, AI, LLM, or “model”.
- Don’t invent causes; describe what’s visible.
""".strip()

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
        temperature=0.3,
        max_output_tokens=170,
    )

    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    # fallback extraction
    try:
        out_parts: List[str] = []
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                tx = getattr(c, "text", None)
                if isinstance(tx, str):
                    out_parts.append(tx)
        return " ".join(out_parts).strip()
    except Exception:
        return ""


def _judge_summary_openai(text: str, facts: Dict[str, Any], title: str, description: str) -> Dict[str, Any]:
    from openai import OpenAI  # lazy import

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    judge_prompt = f"""
You are validating a short insight paragraph for a time-series chart on a professional analytics site.

Chart title: {title}
Chart description: {description}

Facts derived from the plotted series (ground truth):
{json.dumps(facts, ensure_ascii=False, indent=2)}

Candidate paragraph:
{text}

Decide if the paragraph is relevant and faithful to the facts:
- Must be descriptive only (no causal explanations).
- Must not introduce numbers/dates that contradict the facts.
- Should be 3–8 sentences, neutral tone.
- OPTIONAL: Should reference at least one of the provided momentum measures if they exist (YoY/YTD).


Return STRICT JSON ONLY (no markdown, no extra text) with this schema:
{{
  "pass": true/false,
  "reasons": ["..."],
  "fixed_text": "..."  // if pass=false and you can fix it; else ""
}}

Rules for fixed_text:
- Use only the facts above.
- 3-8 sentences, no bullets.
- No meta-AI talk.
""".strip()

    resp = client.responses.create(
        model=OPENAI_JUDGE_MODEL,
        input=judge_prompt,
        temperature=0.0,
        max_output_tokens=240,
    )

    raw = (getattr(resp, "output_text", "") or "").strip()

    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict) or "pass" not in obj:
            return {"pass": False, "reasons": ["Judge returned unexpected JSON shape."], "fixed_text": ""}
        if "reasons" not in obj or not isinstance(obj["reasons"], list):
            obj["reasons"] = ["Judge did not provide reasons list."]
        if "fixed_text" not in obj or not isinstance(obj["fixed_text"], str):
            obj["fixed_text"] = ""
        return obj
    except Exception:
        return {"pass": False, "reasons": ["Judge did not return valid JSON."], "fixed_text": ""}


def _generate_validated_summary(
    facts: Dict[str, Any],
    title: str,
    description: str,
    max_attempts: int = 3,
) -> Tuple[str, Dict[str, Any]]:
    debug: Dict[str, Any] = {"attempts": []}

    for attempt in range(1, max_attempts + 1):
        candidate = _generate_trend_summary_openai(facts, title, description)

        basic_ok, basic_reasons = _basic_summary_checks(candidate, facts)
        judge = _judge_summary_openai(candidate, facts, title, description)

        debug["attempts"].append(
            {
                "attempt": attempt,
                "text": candidate,
                "basic_pass": basic_ok,
                "basic_reasons": basic_reasons,
                "judge": judge,
            }
        )

        if basic_ok and bool(judge.get("pass")):
            return candidate.strip(), debug

        fixed = (judge.get("fixed_text") or "").strip()
        if fixed:
            basic_ok2, basic_reasons2 = _basic_summary_checks(fixed, facts)
            judge2 = _judge_summary_openai(fixed, facts, title, description)

            debug["attempts"].append(
                {
                    "attempt": f"{attempt}.fixed",
                    "text": fixed,
                    "basic_pass": basic_ok2,
                    "basic_reasons": basic_reasons2,
                    "judge": judge2,
                }
            )

            if basic_ok2 and bool(judge2.get("pass")):
                return fixed, debug

    return "", debug


def _last_failure_details(dbg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a compact explanation from the last attempt in dbg,
    to make debugging failures easy without dumping everything.
    """
    attempts = dbg.get("attempts", [])
    if not attempts:
        return {"message": "No attempts recorded."}

    last = attempts[-1]
    judge = last.get("judge", {}) if isinstance(last, dict) else {}
    out: Dict[str, Any] = {
        "last_attempt": last.get("attempt"),
        "last_text": last.get("text"),
        "basic_reasons": last.get("basic_reasons", []),
        "judge_pass": judge.get("pass"),
        "judge_reasons": judge.get("reasons", []),
    }

    fixed = judge.get("fixed_text")
    if isinstance(fixed, str) and fixed.strip():
        out["judge_fixed_text"] = fixed.strip()

    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    # 1) Fetch data
    r = requests.get(DATA_URL, timeout=60)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise ValueError("Expected /data/{id} to return a JSON list")

    # 2) Fetch metadata
    mr = requests.get(META_URL, timeout=60)
    mr.raise_for_status()
    meta_payload = mr.json()
    meta_item = _pick_metadata_item(meta_payload, DATA_ID) or {}

    # 3) Build evergreen payload
    out: Dict[str, Any] = {
        "id": DATA_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meta": {
            "title": meta_item.get("title") or "EU web search trends",
            "description": meta_item.get("description") or "",
            "source": meta_item.get("source") or "",
            "visibility": meta_item.get("visibility") or "public",
            "updated_at": meta_item.get("updated_at") or "",
            "api": {"data_url": DATA_URL, "metadata_url": META_URL},
        },
        "data": data,
    }

    # 4) Facts + hash (always present)
    facts = _compute_trend_facts(data)
    out["meta"]["trend_facts"] = facts
    out["meta"]["data_hash"] = _data_fingerprint(data)

    # 5) Optional AI summary
    if ENABLE_AI_SUMMARY:
        if not os.environ.get("OPENAI_API_KEY"):
            out["meta"]["trend_summary_error"] = "ENABLE_AI_SUMMARY=1 but OPENAI_API_KEY is missing."
        else:
            try:
                summary, dbg = _generate_validated_summary(
                    facts=facts,
                    title=out["meta"]["title"],
                    description=out["meta"]["description"],
                    max_attempts=AI_SUMMARY_MAX_ATTEMPTS,
                )

                if summary:
                    out["meta"]["trend_summary"] = summary
                    out["meta"]["trend_summary_generated_at"] = datetime.now(timezone.utc).isoformat()
                else:
                    out["meta"]["trend_summary_error"] = "Failed validation after retries."
                    out["meta"]["trend_summary_failure"] = _last_failure_details(dbg)

                    if KEEP_AI_DEBUG:
                        out["meta"]["trend_summary_debug"] = dbg
            except Exception as e:
                out["meta"]["trend_summary_error"] = str(e)

    # 6) Write output JSON
    out_dir = os.path.dirname(OUT_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote {OUT_PATH} with {len(data)} rows.")
    if ENABLE_AI_SUMMARY:
        if out["meta"].get("trend_summary"):
            print("AI summary: OK")
        else:
            print("AI summary: NOT PUBLISHED (failed/disabled)")
            failure = out["meta"].get("trend_summary_failure")
            if failure:
                print("AI summary failure details:", json.dumps(failure, ensure_ascii=False))


if __name__ == "__main__":
    main()
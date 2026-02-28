#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests


API_BASE = os.environ.get("API_BASE", "https://furnilytics-api.fly.dev").rstrip("/")
DATA_ID = os.environ.get("DATA_ID", "retail/online/web_search_trends_eu")
OUT_PATH = os.environ.get("OUT_PATH", "docs/web_search_trends_eu.json")

DATA_URL = f"{API_BASE}/data/{DATA_ID}"
META_URL = f"{API_BASE}/metadata"


def _pick_metadata_item(metadata_payload: Dict[str, Any], dataset_id: str) -> Optional[Dict[str, Any]]:
    items = metadata_payload.get("data", [])
    if not isinstance(items, list):
        return None
    for it in items:
        if isinstance(it, dict) and it.get("id") == dataset_id:
            return it
    return None


def main() -> None:
    # 1) Fetch data (list of {date,value})
    r = requests.get(DATA_URL, timeout=60)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise ValueError("Expected /data/{id} to return a JSON list")

    # 2) Fetch metadata catalog and pick item for this dataset
    mr = requests.get(META_URL, timeout=60)
    mr.raise_for_status()
    meta_payload = mr.json()
    meta_item = _pick_metadata_item(meta_payload, DATA_ID) or {}

    # 3) Create a single “evergreen payload” for the page
    out = {
        "id": DATA_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meta": {
            "title": meta_item.get("title") or "EU web search trends",
            "description": meta_item.get("description") or "",
            "source": meta_item.get("source") or "",
            "visibility": meta_item.get("visibility") or "public",
            "updated_at": meta_item.get("updated_at") or "",
            "api": {
                "data_url": DATA_URL,
                "metadata_url": META_URL,
            },
        },
        "data": data,  # [{date, value}, ...]
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote {OUT_PATH} with {len(data)} rows.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List

import requests


API_URL = os.environ.get("API_URL", "https://furnilytics-api.fly.dev/metadata").strip()
OUT_PATH = os.environ.get("OUT_PATH", "docs/dataset_schema.json").strip()


def _to_source_text(source: Any) -> str:
    """Your meta.source can be str or dict. Convert to a short readable string."""
    if source is None:
        return ""
    if isinstance(source, str):
        return source
    if isinstance(source, dict):
        name = source.get("name") or ""
        dataset = source.get("dataset") or ""
        if name and dataset:
            return f"{name} ({dataset})"
        if name:
            return str(name)
        return json.dumps(source, ensure_ascii=False)
    return str(source)


def _normalize_columns(schema: Any) -> List[Dict[str, str]]:
    """Return [{name,type,description}] from your schema object."""
    if not isinstance(schema, dict):
        return []
    cols = schema.get("columns")
    if not isinstance(cols, list):
        return []
    out: List[Dict[str, str]] = []
    for c in cols:
        if not isinstance(c, dict):
            continue
        out.append(
            {
                "name": str(c.get("name") or ""),
                "type": str(c.get("type") or ""),
                "description": str(c.get("description") or ""),
            }
        )
    return out


def main() -> None:
    r = requests.get(API_URL, timeout=60)
    r.raise_for_status()
    payload = r.json()

    items = payload.get("data", [])
    if not isinstance(items, list):
        raise ValueError("Expected payload['data'] to be a list")

    # Build {topics:[{name, subtopics:[{name, tables:[...]}]}]}
    topics_map = defaultdict(lambda: defaultdict(list))

    for it in items:
        if not isinstance(it, dict):
            continue

        dataset_id = str(it.get("id") or "")
        topic = str(it.get("topic") or "other")
        subtopic = str(it.get("subtopic") or "general")
        title = str(it.get("title") or dataset_id)
        description = str(it.get("description") or "")
        source = _to_source_text(it.get("source"))
        columns = _normalize_columns(it.get("schema"))

        # This is the “chip” your UI shows. Keep as the dataset id.
        # (Later you can turn it into /data/{id} if you prefer.)
        path = dataset_id

        topics_map[topic][subtopic].append(
            {
                "title": title,
                "path": path,
                "description": description,
                "source": source,
                "columns": columns,
            }
        )

    topics: List[Dict[str, Any]] = []
    for topic_name in sorted(topics_map.keys()):
        subtopics: List[Dict[str, Any]] = []
        for sub_name in sorted(topics_map[topic_name].keys()):
            tables = sorted(topics_map[topic_name][sub_name], key=lambda x: (x.get("title") or ""))
            subtopics.append({"name": sub_name, "tables": tables})
        topics.append({"name": topic_name, "subtopics": subtopics})

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {"type": "api", "url": API_URL},
        "topics": topics,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote {OUT_PATH} with {len(topics)} topics.")


if __name__ == "__main__":
    main()
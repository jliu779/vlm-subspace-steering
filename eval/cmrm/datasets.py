from __future__ import annotations

import json
from pathlib import Path

from .manifest import ManifestRecord


def load_vlsafe_jsonl(
    path: str | Path,
    coco_root: str | Path | None = None,
    limit: int | None = None,
    id_prefix: str = "vlsafe",
) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and len(records) >= limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            image_id = row.get("image_id") or row.get("image") or row.get("image_path")
            image_path = str(Path(coco_root) / image_id) if coco_root and image_id else image_id
            records.append(
                ManifestRecord(
                    id=str(row.get("id", f"{id_prefix}_{i:06d}")),
                    query=str(row["query"]),
                    image_path=image_path,
                    caption=row.get("caption"),
                    metadata={"source": str(path), **({"reference": row["reference"]} if "reference" in row else {})},
                )
            )
    return records

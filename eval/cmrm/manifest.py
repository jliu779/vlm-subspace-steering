from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class ManifestRecord:
    id: str
    query: str
    image_path: str | None = None
    caption: str | None = None
    metadata: dict = field(default_factory=dict)


def _record_from_dict(row: dict) -> ManifestRecord:
    meta = dict(row.get("metadata") or {})
    known = {"id", "query", "image_path", "caption", "metadata"}
    for key, value in row.items():
        if key not in known:
            meta[key] = value
    return ManifestRecord(
        id=str(row["id"]),
        query=str(row["query"]),
        image_path=row.get("image_path"),
        caption=row.get("caption"),
        metadata=meta,
    )


def read_manifest(path: str | Path, limit: int | None = None) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(_record_from_dict(json.loads(line)))
                if limit and len(records) >= limit:
                    break
    return records


def write_manifest(records: Iterable[ManifestRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            row = asdict(record)
            row = {k: v for k, v in row.items() if v not in (None, {}, [])}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_image_path(record: ManifestRecord, image_root: str | Path | None) -> Path | None:
    if not record.image_path:
        return None
    path = Path(record.image_path)
    return path if path.is_absolute() else Path(image_root or ".") / path


def split_records(
    records: list[ManifestRecord], anchor_ratio: float, seed: int
) -> tuple[list[ManifestRecord], list[ManifestRecord]]:
    if not 0 < anchor_ratio < 1:
        raise ValueError("anchor_ratio must be between 0 and 1")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n_anchor = max(1, round(len(shuffled) * anchor_ratio))
    return shuffled[:n_anchor], shuffled[n_anchor:]

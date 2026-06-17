#!/usr/bin/env python3
"""Optional CLI flags for manifest sharding in generate/baseline scripts."""
from __future__ import annotations

import argparse

from cmrm.manifest import read_manifest


def add_shard_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("manifest sharding")
    group.add_argument(
        "--shard_id",
        type=int,
        default=None,
        help="0-based shard index (use with --num_shards)",
    )
    group.add_argument(
        "--num_shards",
        type=int,
        default=None,
        help="split manifest into this many contiguous chunks",
    )


def load_manifest_records(args: argparse.Namespace):
    shard_id = getattr(args, "shard_id", None)
    num_shards = getattr(args, "num_shards", None)
    if (shard_id is None) ^ (num_shards is None):
        raise SystemExit("error: --shard_id and --num_shards must be set together")
    return read_manifest(
        args.manifest,
        limit=getattr(args, "limit", None),
        shard_id=shard_id,
        num_shards=num_shards,
    )

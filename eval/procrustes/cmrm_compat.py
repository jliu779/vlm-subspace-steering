"""Reuse the CMRM repo's plumbing instead of duplicating it.

We only re-export the pieces that have nothing to do with the steering
geometry: prompt building, LLaVA / Llama loading, manifest IO, last-token
hidden extraction, judge prompt + judge_one, metrics. This way any
correctness fix we make in CMRM (prompt template, judge prompt, manifest
schema) propagates here automatically.
"""
from __future__ import annotations

import sys
from pathlib import Path

_CMRM_ROOT = Path(__file__).resolve().parents[2] / "CMRM"
if str(_CMRM_ROOT) not in sys.path:
    sys.path.insert(0, str(_CMRM_ROOT))

from cmrm.config import load_config, parse_layers, torch_dtype  # noqa: E402,F401
from cmrm.generation import read_jsonl, write_jsonl  # noqa: E402,F401
from cmrm.hidden import (  # noqa: E402,F401
    decoder_hidden_states,
    extract_last_token_hidden,
    prepare_inputs,
)
from cmrm.judging import (  # noqa: E402,F401
    JUDGE_SYSTEM_PROMPT_ACTIONABLE,
    JUDGE_SYSTEM_PROMPT_REFUSAL,
    judge_one,
)
from cmrm.manifest import ManifestRecord, read_manifest  # noqa: E402,F401
from cmrm.models import (  # noqa: E402,F401
    first_model_device,
    get_decoder_layers,
    load_judge,
    load_vlm,
)
from cmrm.prompts import build_prompt  # noqa: E402,F401

#!/usr/bin/env python3
"""Run Procrustes-rotation intervention on the eval manifest using fitted
params from 02_fit_procrustes.py."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

from procrustes.cmrm_compat import (
    first_model_device,
    get_decoder_layers,
    load_config,
    load_vlm,
    parse_layers,
    prepare_inputs,
    read_manifest,
    write_jsonl,
)
from procrustes.hooks import ProcrustesHookManager
from procrustes.io import load_params


def _load_layer_map_json(path: str | None) -> dict[int, float]:
    if not path:
        return {}
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"layer map json must be dict: {path}")
    return {int(k): float(v) for k, v in obj.items()}


def _build_tail_ramp_map(
    target_layers: list[int],
    *,
    base: float,
    tail_n: int,
    non_tail_scale: float,
    tail_scale: float,
) -> dict[int, float]:
    if not target_layers:
        return {}
    ordered = sorted(target_layers)
    tail_n = max(0, min(int(tail_n), len(ordered)))
    tail_layers = set(ordered[-tail_n:]) if tail_n > 0 else set()
    out = {}
    for lid in ordered:
        scale = tail_scale if lid in tail_layers else non_tail_scale
        out[int(lid)] = float(base * scale)
    return out


def _load_cmrm_vectors(path: str | None) -> dict[int, object]:
    if not path:
        return {}
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    vectors = payload.get("vectors", payload)
    if not isinstance(vectors, dict):
        raise ValueError(f"invalid vectors payload: {path}")
    return {int(k): v for k, v in vectors.items()}


def _select_layers_from_rocmrm(
    rocmrm_params_path: str | None,
    *,
    topk: int,
    candidate_layers: list[int],
) -> list[int]:
    if not rocmrm_params_path or topk <= 0:
        return candidate_layers
    import torch

    payload = torch.load(rocmrm_params_path, map_location="cpu", weights_only=False)
    by_layer = payload.get("params_by_layer", {})
    scored = []
    for lid in candidate_layers:
        params = by_layer.get(int(lid))
        if params is None:
            continue
        mean_delta = params.get("mean_delta")
        if mean_delta is None:
            continue
        score = float(mean_delta.norm().item())
        scored.append((score, int(lid)))
    if not scored:
        return candidate_layers
    scored.sort(reverse=True)
    keep = {lid for _s, lid in scored[: min(topk, len(scored))]}
    return [lid for lid in candidate_layers if lid in keep]


_RISK_KEYWORDS = [
    "kill", "harm", "attack", "bomb", "poison", "derail", "smuggle",
    "illegal", "weapon", "chaos", "panic", "damage", "stabbing", "shoot",
    "terror", "terrorist", "hack", "hacking", "malware", "explosive",
    "sabotage", "steal", "trace", "evidence", "bypass", "assault",
    "contaminate", "without getting caught", "without her consent",
    "without consent", "compromising photo", "deface", "vandalize",
    "start a fire", "arson", "threaten", "kidnap", "abuse",
    # extended after L1110 analysis (2026-05-11): rs=0 unsafe queries
    # used indirect harm vocabulary that bypassed the original list.
    "cause maximum", "major accident", "dangerous", "manipulate",
    "deceive", "objectify", "knock down", "hijack", "secretly",
    "less safe", "cause a fire", "scare", "traffic accident",
    "exploit", "demean",
]

_UTILITY_PHRASES = [
    "answer with the option",
    "answer with option",
    "which of the following",
    "choose the correct",
    "context:",
    "question:",
    "options:",
    "describe this image",
    "what is shown",
    "what can you see",
    "what color",
    "how many",
    "where is",
    "identify",
]


def _risk_profile(
    query: str,
    *,
    mode: str,
    low_scale: float,
    high_scale: float,
    keyword_threshold: int,
) -> dict[str, object]:
    keyword_threshold = max(1, int(keyword_threshold))
    if mode == "off":
        return {
            "alpha_scale": 1.0,
            "risk_score": 0,
            "risk_keyword_threshold": keyword_threshold,
            "utility_like": False,
        }
    q = (query or "").lower()
    score = sum(1 for k in _RISK_KEYWORDS if k in q)
    utility_like = any(p in q for p in _UTILITY_PHRASES)
    threshold = 1 if mode == "keyword_any" else keyword_threshold
    if mode == "utility_bypass" and utility_like and score == 0:
        alpha_scale = 0.0
    else:
        alpha_scale = high_scale if score >= threshold else low_scale
    return {
        "alpha_scale": float(alpha_scale),
        "risk_score": int(score),
        "risk_keyword_threshold": int(threshold),
        "utility_like": bool(utility_like),
    }


def _use_benign_orth(scope: str, risk_info: dict[str, object]) -> bool:
    if scope == "off":
        return False
    if scope == "all":
        return True
    score = int(risk_info["risk_score"])
    threshold = int(risk_info["risk_keyword_threshold"])
    utility_like = bool(risk_info["utility_like"])
    if scope == "low_risk":
        return score < threshold
    if scope == "high_risk":
        return score >= threshold
    if scope == "utility":
        return utility_like and score == 0
    raise ValueError(f"unknown benign_orth_scope {scope!r}")


def generate_one(model, processor, record, mode, cfg, image_root=None, generation_overrides=None) -> str:
    import torch

    device = first_model_device(model)
    inputs = prepare_inputs(processor, record, mode, image_root=image_root, device=device)
    gen_cfg = dict(cfg.get("generation", {}))
    if generation_overrides:
        gen_cfg.update(generation_overrides)
    with torch.no_grad():
        gen = model.generate(**inputs, **gen_cfg)
    prompt_len = inputs["input_ids"].shape[-1]
    return processor.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--params", required=True, help="path to fitted Procrustes params .pt")
    parser.add_argument("--mode", default="orig")
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--lambda_mean", type=float)
    parser.add_argument("--rotation_scale", type=float, default=1.0,
                        help="Multiplier on rotation term (set 0 for mean-only).")
    parser.add_argument("--mean_shift_mode", default="full",
                        choices=["full", "projected_rotation", "refusal_projected"],
                        help="'full': lambda_mean * (mu_t-mu_c). 'projected_rotation': "
                             "project mean shift onto per-sample rotation correction; "
                             "no lambda required. 'refusal_projected': project onto refusal "
                             "direction r̂; needs --refusal_dir.")
    parser.add_argument("--refusal_dir", help="refusal direction .pt path")
    parser.add_argument("--layers")
    parser.add_argument(
        "--hook_scope",
        choices=["prefill_only", "all_steps", "prefill_plus_decode_k"],
    )
    parser.add_argument("--decode_k_steps", type=int, default=0)
    parser.add_argument("--trust_region_rho", type=float)
    parser.add_argument("--layerwise_scheme", choices=["none", "tail_ramp"], default="none")
    parser.add_argument("--tail_n", type=int, default=8)
    parser.add_argument("--tail_alpha_scale", type=float, default=1.10)
    parser.add_argument("--non_tail_alpha_scale", type=float, default=0.90)
    parser.add_argument("--tail_lambda_scale", type=float, default=1.00)
    parser.add_argument("--non_tail_lambda_scale", type=float, default=1.00)
    parser.add_argument("--alpha_map_json")
    parser.add_argument("--lambda_map_json")
    parser.add_argument("--cmrm_vectors", help="Optional CMRM vectors .pt for residual fusion")
    parser.add_argument("--residual_alpha", type=float, default=0.0)
    parser.add_argument("--residual_sign", type=int, default=1)
    parser.add_argument(
        "--risk_gating",
        choices=["off", "keyword", "keyword_any", "utility_bypass"],
        default="off",
    )
    parser.add_argument("--risk_low_scale", type=float, default=0.85)
    parser.add_argument("--risk_high_scale", type=float, default=1.00)
    parser.add_argument(
        "--risk_keyword_threshold",
        type=int,
        default=2,
        help="minimum matched risk keywords needed for high-risk scaling",
    )
    parser.add_argument("--rocmrm_params", help="Optional RoCMRM params .pt for layer mask")
    parser.add_argument("--rocmrm_layer_topk", type=int, default=0)
    parser.add_argument("--benign_paired", help=(
        "Optional path to benign paired_hidden.pt. When provided, fit a "
        "rank-`benign_k` per-layer subspace from H_t in this file and "
        "orthogonalize the Procrustes correction against it (Variant A)."
    ))
    parser.add_argument("--benign_k", type=int, default=16,
                        help="rank of benign-task subspace G")
    parser.add_argument(
        "--benign_orth_scope",
        choices=["all", "low_risk", "high_risk", "utility", "off"],
        default="all",
        help=(
            "when --benign_paired is set, choose which records receive "
            "G-perp orthogonalization"
        ),
    )
    parser.add_argument(
        "--utility_guard_paired",
        help=(
            "Optional paired_hidden.pt used to fit a utility subspace guard. "
            "When active, records/layers whose hidden state strongly projects "
            "onto this subspace have their whole Procrustes delta scaled down."
        ),
    )
    parser.add_argument("--utility_guard_k", type=int, default=16)
    parser.add_argument("--utility_guard_threshold", type=float)
    parser.add_argument("--utility_guard_scale", type=float, default=0.0)
    parser.add_argument(
        "--lda_guard",
        help=(
            "Path to lda_guard_*.pt produced by fitting LDA between VLSafe and "
            "utility (ScienceQA) hidden states. When active, the Procrustes delta "
            "is scaled by --lda_guard_scale for samples that score as "
            "utility-like (LDA score > 0), while VLSafe-like samples (score < 0) "
            "receive full correction."
        ),
    )
    parser.add_argument("--lda_guard_scale", type=float, default=0.0)
    parser.add_argument(
        "--mean_shift_gate_threshold",
        type=float,
        help=(
            "If set, scale the global mean-shift term by a sigmoid of the "
            "in-subspace projection ratio. Benign-like inputs (low ratio) "
            "receive no mean shift; harmful-like inputs (high ratio) get "
            "the full shift. Geometric replacement for keyword gating."
        ),
    )
    parser.add_argument("--mean_shift_gate_temp", type=float, default=0.05)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max_new_tokens", type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    proc_cfg = cfg.get("procrustes", {})
    alpha = args.alpha if args.alpha is not None else float(proc_cfg.get("alpha", 0.5))
    lambda_mean = (
        args.lambda_mean if args.lambda_mean is not None
        else float(proc_cfg.get("lambda_mean", 0.0))
    )
    layers_spec = args.layers or proc_cfg.get("layers", "all")
    hook_scope = args.hook_scope or proc_cfg.get("hook_scope", "prefill_only")

    payload = load_params(args.params)
    params_by_layer = payload["params_by_layer"]
    image_root = cfg.get("paths", {}).get("image_root")
    records = read_manifest(args.manifest, limit=args.limit)

    benign_subspace_by_layer = {}
    if args.benign_paired:
        import torch as _torch
        from procrustes.subspace import fit_benign_subspace
        benign_payload = _torch.load(args.benign_paired, map_location="cpu", weights_only=False)
        H_benign_by_layer = {int(l): v["H_t"].float() for l, v in benign_payload["by_layer"].items()}
        benign_subspace_by_layer = fit_benign_subspace(
            H_benign_by_layer, k=args.benign_k, centered=True,
        )
        print(f"loaded benign anchor: {len(benign_subspace_by_layer)} layers, "
              f"k={args.benign_k}, anchor={benign_payload['meta'].get('anchor_manifest', '?')}")

    utility_guard_by_layer = {}
    if args.utility_guard_paired:
        import torch as _torch
        from procrustes.subspace import fit_benign_subspace_with_mean
        utility_payload = _torch.load(args.utility_guard_paired, map_location="cpu", weights_only=False)
        H_utility_by_layer = {int(l): v["H_t"].float() for l, v in utility_payload["by_layer"].items()}
        utility_guard_by_layer = fit_benign_subspace_with_mean(
            H_utility_by_layer, k=args.utility_guard_k, centered=True,
        )
        print(
            f"loaded utility guard: {len(utility_guard_by_layer)} layers, "
            f"k={args.utility_guard_k}, manifest={utility_payload['meta'].get('manifest', '?')}"
        )

    lda_guard_by_layer = {}
    if args.lda_guard:
        import torch as _torch
        lda_payload = _torch.load(args.lda_guard, map_location="cpu", weights_only=False)
        lda_guard_by_layer = {int(l): v for l, v in lda_payload["by_layer"].items()}
        print(
            f"loaded LDA guard: {len(lda_guard_by_layer)} layers, "
            f"scale={args.lda_guard_scale}, "
            f"meta={lda_payload['meta'].get('label', '?')}"
        )

    model, processor = load_vlm(cfg)
    decoder_layers = get_decoder_layers(model)
    requested = parse_layers(layers_spec, len(decoder_layers))
    target_layers = [l for l in requested if int(l) in params_by_layer]
    target_layers = _select_layers_from_rocmrm(
        args.rocmrm_params,
        topk=args.rocmrm_layer_topk,
        candidate_layers=target_layers,
    )
    if not target_layers:
        raise ValueError("requested layers do not intersect fitted params")

    alpha_by_layer = _load_layer_map_json(args.alpha_map_json)
    lambda_by_layer = _load_layer_map_json(args.lambda_map_json)
    if args.layerwise_scheme == "tail_ramp":
        alpha_by_layer = _build_tail_ramp_map(
            target_layers,
            base=alpha,
            tail_n=args.tail_n,
            non_tail_scale=args.non_tail_alpha_scale,
            tail_scale=args.tail_alpha_scale,
        )
        lambda_by_layer = _build_tail_ramp_map(
            target_layers,
            base=lambda_mean,
            tail_n=args.tail_n,
            non_tail_scale=args.non_tail_lambda_scale,
            tail_scale=args.tail_lambda_scale,
        )
    residual_vectors_by_layer = _load_cmrm_vectors(args.cmrm_vectors)

    refusal_dir_by_layer = {}
    if args.refusal_dir:
        import torch as _torch
        rd_payload = _torch.load(args.refusal_dir, map_location="cpu", weights_only=False)
        refusal_dir_by_layer = {int(l): v for l, v in rd_payload["refusal_dir_by_layer"].items()}
        print(f"loaded refusal_dir for {len(refusal_dir_by_layer)} layers from {args.refusal_dir}")

    rows = []
    generation_overrides = {}
    if args.max_new_tokens is not None:
        generation_overrides["max_new_tokens"] = args.max_new_tokens
    if benign_subspace_by_layer:
        variant_tag = "_orthG" if args.benign_orth_scope == "all" else f"_orthG{args.benign_orth_scope}"
    else:
        variant_tag = ""
    scheme_tag = "" if args.layerwise_scheme == "none" else f"_lw{args.layerwise_scheme}"
    hook_tag = (
        "_pk"
        if hook_scope == "prefill_plus_decode_k"
        else ("_pf" if hook_scope == "prefill_only" else "_all")
    )
    trust_tag = "" if args.trust_region_rho is None else f"_rho{args.trust_region_rho}"
    residual_tag = "" if args.residual_alpha <= 0 else f"_res{args.residual_alpha}"
    guard_tag = (
        ""
        if not utility_guard_by_layer
        else f"_uguard{args.utility_guard_k}t{args.utility_guard_threshold}x{args.utility_guard_scale}"
    )
    lda_tag = "" if not lda_guard_by_layer else f"_ldag{args.lda_guard_scale}"
    method_tag = (
        f"procrustes_{payload['meta']['centering_mode']}_"
        f"{payload['meta']['subspace_mode']}_k{payload['meta']['k']}"
        f"{variant_tag}{scheme_tag}{hook_tag}{trust_tag}{residual_tag}{guard_tag}{lda_tag}"
    )
    for record in tqdm(records, desc=method_tag):
        risk_info = _risk_profile(
            record.query,
            mode=args.risk_gating,
            low_scale=args.risk_low_scale,
            high_scale=args.risk_high_scale,
            keyword_threshold=args.risk_keyword_threshold,
        )
        alpha_scale = float(risk_info["alpha_scale"])
        benign_orth_applied = bool(
            benign_subspace_by_layer and _use_benign_orth(args.benign_orth_scope, risk_info)
        )
        with ProcrustesHookManager(
            decoder_layers,
            params_by_layer,
            target_layers,
            alpha=alpha,
            lambda_mean=lambda_mean,
            rotation_scale=args.rotation_scale,
            mean_shift_mode=args.mean_shift_mode,
            refusal_dir_by_layer=refusal_dir_by_layer if refusal_dir_by_layer else None,
            hook_scope=hook_scope,
            decode_k_steps=args.decode_k_steps,
            alpha_by_layer=alpha_by_layer,
            lambda_by_layer=lambda_by_layer,
            alpha_scale=alpha_scale,
            trust_region_rho=args.trust_region_rho,
            residual_vectors_by_layer=residual_vectors_by_layer,
            residual_alpha=args.residual_alpha,
            residual_sign=args.residual_sign,
            benign_subspace_by_layer=(
                benign_subspace_by_layer if benign_orth_applied else None
            ),
            utility_guard_by_layer=utility_guard_by_layer if utility_guard_by_layer else None,
            utility_guard_threshold=args.utility_guard_threshold,
            utility_guard_scale=args.utility_guard_scale,
            lda_guard_by_layer=lda_guard_by_layer if lda_guard_by_layer else None,
            lda_guard_scale=args.lda_guard_scale,
            mean_shift_gate_threshold=args.mean_shift_gate_threshold,
            mean_shift_gate_temp=args.mean_shift_gate_temp,
        ):
            response = generate_one(
                model,
                processor,
                record,
                args.mode,
                cfg,
                image_root=image_root,
                generation_overrides=generation_overrides,
            )
        rows.append({
            "id": record.id,
            "query": record.query,
            "method": method_tag,
            "input_mode": args.mode,
            "model_id": cfg["model"]["target_model_id"],
            "response": response,
            "alpha": alpha,
            "lambda_mean": lambda_mean,
            "k": payload["meta"]["k"],
            "centering_mode": payload["meta"]["centering_mode"],
            "subspace_mode": payload["meta"]["subspace_mode"],
            "hook_scope": hook_scope,
            "decode_k_steps": args.decode_k_steps,
            "trust_region_rho": args.trust_region_rho,
            "layerwise_scheme": args.layerwise_scheme,
            "risk_gating": args.risk_gating,
            "alpha_scale": alpha_scale,
            "risk_score": risk_info["risk_score"],
            "risk_keyword_threshold": risk_info["risk_keyword_threshold"],
            "utility_like": risk_info["utility_like"],
            "benign_orth_scope": args.benign_orth_scope,
            "benign_orth_applied": benign_orth_applied,
            "utility_guard_paired": args.utility_guard_paired,
            "utility_guard_k": args.utility_guard_k if utility_guard_by_layer else None,
            "utility_guard_threshold": args.utility_guard_threshold,
            "utility_guard_scale": args.utility_guard_scale if utility_guard_by_layer else None,
            "lda_guard": args.lda_guard,
            "lda_guard_scale": args.lda_guard_scale if lda_guard_by_layer else None,
            "residual_alpha": args.residual_alpha,
        })
    write_jsonl(rows, args.out)
    print(f"wrote {len(rows)} {method_tag} rows to {args.out}")


if __name__ == "__main__":
    main()

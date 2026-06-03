from __future__ import annotations

MULTIMODAL_MODES = {"orig", "blank", "noise"}
TEXT_MODES = {"caption", "query_only"}
ALL_MODES = MULTIMODAL_MODES | TEXT_MODES


def needs_image(mode: str) -> bool:
    if mode not in ALL_MODES:
        raise ValueError(f"unknown input mode {mode!r}")
    return mode in MULTIMODAL_MODES


def build_prompt(query: str, mode: str, caption: str | None = None) -> str:
    if mode not in ALL_MODES:
        raise ValueError(f"unknown input mode {mode!r}")
    if mode in MULTIMODAL_MODES:
        return f"USER: <image>\n{query}\nASSISTANT:"
    if mode == "caption":
        text = f"{caption}\n{query}" if caption else query
        return f"USER: {text}\nASSISTANT:"
    return f"USER: {query}\nASSISTANT:"

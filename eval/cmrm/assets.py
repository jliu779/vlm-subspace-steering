from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

MODEL_ROOT = Path("/hub/huggingface/models")
DATASET_ROOT = Path("/hub/huggingface/datasets")


def _repo_candidates(repo_id: str) -> list[Path]:
    org, name = repo_id.split("/", 1)
    aliases = [org]
    if org == "meta-llama":
        aliases.append("meta")
    return [Path(alias) / name for alias in aliases]


def local_model_path(repo_id: str, root: str | Path = MODEL_ROOT) -> Path | None:
    root = Path(root)
    for rel in _repo_candidates(repo_id):
        path = root / rel
        if path.exists():
            return path
    hf_cache = root / f"models--{repo_id.replace('/', '--')}"
    if hf_cache.exists():
        return hf_cache
    return None


def local_dataset_path(repo_id: str, root: str | Path = DATASET_ROOT) -> Path | None:
    root = Path(root)
    org, name = repo_id.split("/", 1)
    candidates = [
        root / org / name,
        root / f"{org}_{name}",
        root / f"{org}___{name.lower()}",
        root / f"datasets--{repo_id.replace('/', '--')}",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def resolve_model_id(repo_id: str, root: str | Path = MODEL_ROOT) -> str:
    local = local_model_path(repo_id, root)
    return str(local) if local else repo_id


def resolve_dataset_name(repo_id: str, root: str | Path = DATASET_ROOT) -> str:
    local = local_dataset_path(repo_id, root)
    return str(local) if local else repo_id


def download_command(repo_id: str, repo_type: str, local_dir: str | Path) -> list[str]:
    project_root = Path(__file__).resolve().parents[1]
    tool = shutil.which("huggingface-download") or _existing_executable(project_root / ".venv" / "bin" / "huggingface-download")
    if tool:
        cmd = [tool, repo_id, "--local-dir", str(local_dir)]
        if repo_type == "dataset":
            cmd.extend(["--repo-type", "dataset"])
        return cmd

    hf = (
        shutil.which("hf")
        or _existing_executable(project_root / ".venv" / "bin" / "hf")
        or shutil.which("huggingface-cli")
        or _existing_executable(project_root / ".venv" / "bin" / "huggingface-cli")
    )
    if hf:
        cmd = [hf, "download", repo_id, "--local-dir", str(local_dir)]
        if repo_type == "dataset":
            cmd.extend(["--repo-type", "dataset"])
        return cmd

    raise FileNotFoundError("missing huggingface-download, hf, or huggingface-cli")


def _existing_executable(path: Path) -> str | None:
    return str(path) if path.exists() and os.access(path, os.X_OK) else None


def ensure_repo(repo_id: str, repo_type: str, root: str | Path, dry_run: bool = False) -> Path:
    root = Path(root)
    existing = local_dataset_path(repo_id, root) if repo_type == "dataset" else local_model_path(repo_id, root)
    if existing:
        return existing
    org, name = repo_id.split("/", 1)
    local_dir = root / org / name
    if dry_run:
        return local_dir
    cmd = download_command(repo_id, repo_type, local_dir)
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("HF_HOME", str(root / ".cache" / "huggingface"))
    subprocess.run(cmd, check=True, env=env)
    return local_dir

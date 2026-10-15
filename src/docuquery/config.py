"""Central settings. Everything configurable comes from environment variables.

Why: secrets (API key) must never be in code, and model names should be
swappable without code changes (so eval results can record exactly what ran).
"""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(RuntimeError):
    """Raised when a required setting is missing or inconsistent."""


@dataclass(frozen=True)
class Settings:
    api_key: str
    embed_model: str
    gen_model: str
    judge_model: str      # model used ONLY by the evaluation judges
    chunk_size: int       # characters per chunk
    chunk_overlap: int    # characters shared between neighbouring chunks
    top_k: int            # passages given to the generator
    pool: int             # candidates fetched before fusion / reranking
    pdf_dir: Path
    index_dir: Path


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(
            f"Missing environment variable {name}. "
            "Copy .env.example to .env and fill it in."
        )
    return value


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


def load_settings(load_env_file: bool = True) -> Settings:
    """Read settings from the environment (and .env if present)."""
    if load_env_file:
        # Does not override variables that are already set in the real environment.
        load_dotenv(PROJECT_ROOT / ".env")
    gen_model = _require("GEN_MODEL")
    return Settings(
        api_key=_require("GEMINI_API_KEY"),
        embed_model=_require("EMBED_MODEL"),
        gen_model=gen_model,
        judge_model=os.getenv("JUDGE_MODEL", "").strip() or gen_model,
        chunk_size=_int("CHUNK_SIZE", 1000),
        chunk_overlap=_int("CHUNK_OVERLAP", 200),
        top_k=_int("TOP_K", 5),
        pool=_int("CANDIDATE_POOL", 20),
        pdf_dir=PROJECT_ROOT / "data" / "pdfs",
        index_dir=PROJECT_ROOT / "data" / "index",
    )

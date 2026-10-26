"""Configuration for the ingestion pipeline.

Every caller (upload route, CLI, tests) uses this one object, so behaviour does
not drift between paths. External vision is off by default; the application
boundary must turn it on explicitly.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class ParseConfig:
    # Privacy. No page image leaves the process unless this is true.
    allow_external: bool = field(
        default_factory=lambda: _env_bool("ALLOW_EXTERNAL_VISION", False)
    )
    provider_data_collection_deny: bool = True
    provider_zero_retention: bool = False  # opt-in: not every model has a ZDR route

    # Vision.
    vision_model: str | None = None
    vision_max_tokens: int = 4096
    vision_retries: int = 3
    vision_timeout: float = 120.0
    vision_backoff: float = 1.5
    vision_concurrency: int = 2
    vision_budget_usd: float | None = None

    # OCR and scan assessment.
    scan_dpi: int = 200
    dense_dpi: int = 300
    scan_text_coverage: float = 0.02
    scan_garbled_ratio: float = 0.20
    scan_min_alnum: int = 20
    scan_image_coverage: float = 0.10
    ocr_agreement_threshold: float = 0.85

    # Figures. Deliberately low: stamps, signatures and small diagrams are tiny.
    figure_min_area: float = 0.0005
    figure_min_side: float = 12.0
    figure_prompt_max_tokens: int = 4096

    # Truncation handling.
    tile_max_depth: int = 2
    tile_overlap: float = 0.08
    tile_min_words: int = 8

    # Tables.
    extract_tables: bool = True
    table_empty_marker: str = "(empty)"

    # DOCX. LibreOffice is used only when it is installed.
    docx_via_pdf: bool = True

    # Per-page isolation and budgets.
    page_timeout: float = 300.0
    max_pages: int | None = None

    # Artifacts and checkpointing.
    image_dir: Path | None = None
    store: Any = None
    store_root: Path | None = None

    def fingerprint(self) -> str:
        """Stable hash of behaviour-affecting settings, for cache keys."""
        payload = asdict(self)
        for key in ("store", "store_root", "image_dir"):
            payload.pop(key, None)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

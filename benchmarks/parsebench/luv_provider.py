"""ParseBench provider for luvparse.

Registers provider ``luv`` and pipelines:

  luv_local     structure view, no external calls
  luv_vision    + vision charts-to-tables and OCR of text-less pages
  luv_notags    ablation: header/footer text left out of the markdown tags
  luv_plain     ablation: engine output only (no own tables, no sup/sub/u/mark)

Layout is scored by ParseBench's own pymupdf4llm adapter: raw_output pages carry a
``page_boxes`` list, which is what that adapter keys on.
"""

from __future__ import annotations

import os

import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from parse_bench.inference.pipelines import register_pipeline  # noqa: E402
from parse_bench.inference.providers.base import (  # noqa: E402
    Provider,
    ProviderPermanentError,
)
from parse_bench.inference.providers.registry import register_provider  # noqa: E402
from parse_bench.schemas.parse_output import (  # noqa: E402
    LayoutItemIR,
    LayoutSegmentIR,
    PageIR,
    ParseLayoutPageIR,
    ParseOutput,
)
from parse_bench.schemas.pipeline import PipelineSpec  # noqa: E402
from parse_bench.schemas.pipeline_io import (  # noqa: E402
    InferenceRequest,
    InferenceResult,
    RawInferenceResult,
)
from parse_bench.schemas.product import ProductType  # noqa: E402

from parse_bench.evaluation.layout_adapters.adapters import PyMuPDF4LLMLayoutAdapter  # noqa: E402
from parse_bench.evaluation.layout_adapters.registry import register_layout_adapter  # noqa: E402

from luvparse.structured import StructuredConfig, structure_page  # noqa: E402



@register_layout_adapter("luv", priority=95)
class LuvLayoutAdapter(PyMuPDF4LLMLayoutAdapter):
    """Same IR as the pymupdf4llm provider, selected by this provider's own key.

    ParseBench picks the layout adapter by provider name first, so without this
    entry Layout and Visual Grounding silently score 0 for provider ``luv``.
    """


_LOCK = threading.Lock()  # PyMuPDF and the layout model are not thread safe
_SPEND = {"usd": 0.0, "stopped": False}  # OpenRouter spend this process, with a hard cap
_BUDGET = float(os.environ.get("LUV_BUDGET_USD", "0.58"))


@register_provider("luv")
class LuvProvider(Provider):
    def __init__(self, provider_name: str, base_config: dict[str, Any] | None = None):
        super().__init__(provider_name, base_config)

    def _config(self) -> StructuredConfig:
        cfg = self.base_config or {}
        fields = StructuredConfig.__dataclass_fields__
        return StructuredConfig(**{k: v for k, v in cfg.items() if k in fields})

    @staticmethod
    def _vision(wanted: bool):
        if not wanted:
            return None
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")  # OPENROUTER_API_KEY lives here
        from luvparse.config import ParseConfig
        from luvparse.vision import vision_from_env

        client = vision_from_env(None, config=ParseConfig(allow_external=True))
        if client is None:
            raise ProviderPermanentError("OPENROUTER_API_KEY is not set in .env")
        return client

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.product_type != ProductType.PARSE:
            raise ProviderPermanentError(f"luv supports PARSE only, got {request.product_type}")
        path = Path(request.source_file_path)
        if not path.exists():
            raise ProviderPermanentError(f"File not found: {path}")

        import pymupdf

        config = self._config()
        vision = self._vision(config.chart_tables or config.ocr_scanned or config.vision_tables)
        if vision is not None and _SPEND["usd"] >= _BUDGET:
            if not _SPEND["stopped"]:
                _SPEND["stopped"] = True
                print(f"\n[luv] budget ${_BUDGET:.2f} reached (spent ${_SPEND['usd']:.3f}); remaining files run without vision\n", flush=True)
            vision = None
        started = datetime.now()
        pages: list[dict[str, Any]] = []
        try:
            with _LOCK:
                document = pymupdf.open(str(path))
                try:
                    for index in range(len(document)):
                        page = structure_page(document, index, config, vision)
                        data = page.to_dict()
                        data["page_index"] = index
                        data["page_boxes"] = [
                            {"class": it["label"], "bbox": it["bbox"]} for it in page.items
                        ]
                        pages.append(data)
                finally:
                    document.close()
        except Exception as exc:
            raise ProviderPermanentError(f"luv parser error: {exc!r}") from exc
        finally:
            if vision is not None:
                try:
                    _SPEND["usd"] += sum(float(getattr(r, "cost_usd", 0) or 0) for r in vision.ledger.records)
                    Path("spend_so_far.txt").write_text(f"{_SPEND['usd']:.4f} USD (cap {_BUDGET})\n")
                except Exception:
                    pass
            if vision is not None and hasattr(vision, "close"):
                vision.close()

        done = datetime.now()
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type=request.product_type,
            raw_output={"pages": pages, "num_pages": len(pages)},
            started_at=started,
            completed_at=done,
            latency_in_ms=int((done - started).total_seconds() * 1000),
        )

    @staticmethod
    def _layout_page(data: dict[str, Any]) -> ParseLayoutPageIR | None:
        width, height = float(data.get("width") or 0), float(data.get("height") or 0)
        if width <= 0 or height <= 0:
            return None
        items = []
        for it in data.get("items", []):
            box = it["bbox"]
            segment = LayoutSegmentIR(
                x=box["x"],
                y=box["y"],
                w=box["w"],
                h=box["h"],
                label=it["label"],
                start_index=it["start_index"],
                end_index=it["end_index"],
            )
            items.append(
                LayoutItemIR(
                    type=it["type"],
                    md=it["md"],
                    html=it.get("html", ""),
                    value=it["md"],
                    bbox=segment,
                    layout_segments=[segment],
                )
            )
        return ParseLayoutPageIR(
            page_number=int(data["page_number"]),
            width=width,
            height=height,
            md=data["markdown"],
            text=data["markdown"],
            page_header_markdown=data.get("page_header_markdown") or None,
            page_footer_markdown=data.get("page_footer_markdown") or None,
            printed_page_number=data.get("printed_page_number") or None,
            items=items,
        )

    def normalize(self, raw: RawInferenceResult) -> InferenceResult:
        pages, layouts, texts = [], [], []
        for data in raw.raw_output.get("pages", []):
            markdown = data.get("markdown", "") or ""
            pages.append(PageIR(page_index=data.get("page_index", 0), markdown=markdown))
            texts.append(markdown)
            layout = self._layout_page(data)
            if layout is not None:
                layouts.append(layout)
        output = ParseOutput(
            task_type="parse",
            example_id=raw.request.example_id,
            pipeline_name=raw.pipeline_name,
            pages=pages,
            layout_pages=layouts,
            markdown="\n\n".join(texts),
        )
        return InferenceResult(
            request=raw.request,
            pipeline_name=raw.pipeline_name,
            product_type=raw.product_type,
            raw_output=raw.raw_output,
            output=output,
            started_at=raw.started_at,
            completed_at=raw.completed_at,
            latency_in_ms=raw.latency_in_ms,
        )


def register_pipelines() -> None:
    def spec(name: str, **config: Any) -> None:
        register_pipeline(
            PipelineSpec(
                pipeline_name=name,
                provider_name="luv",
                product_type=ProductType.PARSE,
                config=config,
            )
        )

    spec("luv_local")
    spec("luv_vision", chart_tables=True, ocr_scanned=True, vision_tables=True)
    spec("luv_vision_nt", chart_tables=True, ocr_scanned=True, vision_tables=True, promote_titles=False)
    spec("luv_vision_old", chart_tables=True, ocr_scanned=True)
    spec("luv_vision_alltables", chart_tables=True, ocr_scanned=True, vision_tables=True, vision_tables_all=True)
    spec("luv_notags", emit_page_tags=False)
    spec("luv_plain", own_tables=False, inline_extras=False, clean_text=False)


register_pipelines()

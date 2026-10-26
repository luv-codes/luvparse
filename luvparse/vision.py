"""OpenRouter vision client for OCR and figure understanding.

Reliability rules, all enforced here rather than left to callers:

* ``max_tokens`` is always set and ``finish_reason`` is always read.
* A response that hits the token limit is retried as overlapping strips and
  stitched. A page is never accepted as complete when it was cut off.
* An empty OCR response is checked against image ink density. An image that is
  clearly not blank does not silently become "no text".
* Retries use one shared HTTP connection, exponential backoff with jitter, and
  honour ``Retry-After``.
* Every call is written to the shared UsageLedger with no page content.

Privacy: ``allow_external`` must be true before any image is sent. The
pipeline also gates client creation, so this is a second line of defence.
"""

from __future__ import annotations

import base64
import io
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

from luvparse.usage import UsageLedger

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-2.5-flash"

OCR_PROMPTS = {
    "furniture": (
        "This image is a thin strip from the very top or bottom of a scanned page. "
        "Transcribe every word in it exactly as printed, one printed line per line. "
        "Any word or line with a line drawn through it (struck out) must be written "
        "with the whole word between double tildes, like ~~WORD~~. Keep capital "
        "letters as printed. Output only the transcription, nothing else, no code fences. "
        "If the strip holds no text, output nothing."
    ),
    "markdown": (
        "Transcribe this page to Markdown exactly as printed, in reading order. "
        "Use # headings for titles and section headings. Mark **bold**, *italic*, "
        "<u>underlined</u> text, ~~struck-through~~ text, <mark>highlighted</mark> "
        "text, <sup>superscripts</sup> and <sub>subscripts</sub>. Put running page "
        "headers inside <page_header>...</page_header> and running footers inside "
        "<page_footer>...</page_footer>, one printed line per line, with formatting "
        "kept inside the tags. Write tables as HTML <table> with <th> header cells "
        "and rowspan/colspan for merged cells. Do not add commentary or code fences. "
        "Do not guess unreadable text; write [unclear]."
    ),
    "default": (
        "Transcribe every word of text in this image exactly as printed, "
        "preserving line breaks, table rows and punctuation. Output only the "
        "transcription. Do not summarise. If there is no legible text, output "
        "an empty string."
    ),
    "dense": (
        "Transcribe every character in this image exactly as printed, including "
        "headers, footers, footnotes, stamps and small print. Preserve reading "
        "order and line breaks. Output only the transcription, with no commentary."
    ),
    "table": (
        "Transcribe the table in this image as rows of text. Preserve every cell, "
        "including empty cells, and repeat the column headers for each row. Output "
        "only the transcription."
    ),
}

FIGURE_PROMPTS = {
    "table_html": (
        "This image is one or more tables cropped from a document page. Reply with "
        "HTML only: one <table> per separate table, in reading order, nothing else. "
        "Copy every cell's text exactly as printed, with its numbers, signs, "
        "symbols, currency marks and percent signs; do not summarise, round, "
        "translate or add words. Use <th> for header cells and <td> for the rest. "
        "Use rowspan and colspan wherever a cell spans rows or columns, so every "
        "row has the same number of columns. Keep empty cells as empty <td></td>. "
        "A title or caption printed above a table is not part of it; leave it out. "
        "Do not drop any row, even in a long table, and do not stop early. "
        "Replace runs of dots or dashes used as leaders with nothing."
    ),
    "chart_table": (
        "This is a chart. Reply with ONE HTML table and nothing else. Put the "
        "category or x-axis labels down the first column, exactly as printed "
        "(keep ranges such as 2020/2021 as written), and one column per series. "
        "The series names come from the legend, the colour key, or the group "
        "labels printed beside the bars (for example Favorable attitude and "
        "Unfavorable attitude, or OECD and Non-OECD); use those exact names in "
        "the header row, with the unit if shown. If a category has a value in "
        "only some series, leave the other cells empty. For a pie or donut "
        "chart use two columns, slice name and value, adding a percent column "
        "when shown. Copy every printed number exactly, including the sign of "
        "negative values; read each number against its own bar or point and "
        "never swap numbers between neighbouring bars. If values are not "
        "printed, read them off the axis to the precision the gridlines allow. "
        "Use <th> for header cells. Include the chart title as a <caption>."
    ),
    "figure": (
        "Describe this figure from a document. State what it shows and "
        "every legible label, axis, unit, legend entry and value. Then transcribe "
        "verbatim any text inside the image. Do not speculate beyond what is visible."
    ),
    "chart": (
        "This is a chart. Transcribe the title, both axis labels and units, the "
        "legend, and every legible data value or category label exactly. Then "
        "state in one sentence what the chart shows. Do not omit values."
    ),
    "table": (
        "This is a table image. Transcribe every row exactly, repeating the column "
        "headers for each row. Preserve numbers, units and empty cells. Then state "
        "in one sentence what the table contains."
    ),
    "stamp": (
        "This is a stamp, signature or handwritten note. Transcribe every legible "
        "word and number exactly, including dates and names. Mark any word you "
        "cannot read as [unclear]. Do not guess."
    ),
    "handwriting": (
        "Transcribe the handwriting exactly, line by line. Preserve spelling, "
        "numbers and abbreviations. Mark uncertain words as [unclear]. Do not "
        "paraphrase."
    ),
}

_RETRY_STATUS = {429, 500, 502, 503, 504}
_REFUSAL = re.compile(
    r"^\s*(?:i(?:'m| am) (?:sorry|unable)|i cannot|i can't|the image (?:shows|contains)|"
    r"this image (?:shows|contains)|as an ai)\b",
    re.I,
)
_TRUNCATED_TAIL = re.compile(
    r"(?:\b(?:and|or|the|of|to|in|with|for|from|a|an|is|are|was|were)|\d+[.,]?)\s*$",
    re.I,
)


class VisionError(RuntimeError):
    """A vision call failed. Never carries document text."""


@dataclass(frozen=True)
class VisionResult:
    text: str
    finish_reason: str = ""
    tiles: int = 1
    model: str = ""
    warnings: tuple[str, ...] = ()
    usage: dict[str, Any] = field(default_factory=dict)


def _ink_density(image: bytes) -> float:
    """Fraction of dark pixels. Used to tell a blank response from a blank image."""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(image)) as picture:
            grey = picture.convert("L")
            histogram = grey.histogram()
        total = sum(histogram)
        if not total:
            return 0.0
        dark = sum(histogram[:200])
        return dark / total
    except Exception:
        return 0.0


def _tile_strips(image: bytes, overlap_fraction: float, strips: int = 2) -> list[bytes]:
    """Split an image into horizontal strips with overlap, as PNG bytes."""
    from PIL import Image

    with Image.open(io.BytesIO(image)) as picture:
        picture = picture.convert("RGB")
        width, height = picture.size
        strip_height = max(1, height // strips)
        overlap = max(1, int(strip_height * max(0.0, min(overlap_fraction, 0.5))))
        tiles: list[bytes] = []
        top = 0
        while top < height:
            bottom = min(height, top + strip_height + overlap)
            tile = picture.crop((0, top, width, bottom))
            buffer = io.BytesIO()
            tile.save(buffer, format="PNG")
            tiles.append(buffer.getvalue())
            if bottom >= height:
                break
            top = max(0, bottom - overlap)
        return tiles


def _stitch(parts: list[str]) -> str:
    """Join tile transcriptions, removing overlapping lines and words."""
    result: list[str] = []
    for part in parts:
        if not part.strip():
            continue
        if not result:
            result = part.splitlines()
            continue
        incoming = part.splitlines()
        matched = 0
        for size in range(min(len(result), len(incoming)), 0, -1):
            if [line.strip() for line in result[-size:]] == [
                line.strip() for line in incoming[:size]
            ]:
                matched = size
                break
        if matched:
            result.extend(incoming[matched:])
            continue
        # Fall back to word overlap when line boundaries do not line up.
        previous = " ".join(result)
        words_previous = previous.split()
        words_incoming = part.split()
        overlap = 0
        for size in range(min(len(words_previous), len(words_incoming), 60), 0, -1):
            if words_previous[-size:] == words_incoming[:size]:
                overlap = size
                break
        if overlap:
            result.append(" ".join(words_incoming[overlap:]))
        else:
            result.append(part)
    return "\n".join(line for line in result if line.strip())


def _looks_like_loop(text: str) -> bool:
    """Detect degenerate repetition, a common model failure."""
    compact = " ".join(text.split())
    if len(compact) < 80:
        return False
    for size in (20, 40, 80):
        if len(compact) < size * 4:
            continue
        window = compact[:size]
        if compact.count(window) >= 4 and len(set(compact.split())) < 10:
            return True
    words = compact.split()
    if len(words) >= 40:
        most = max(words.count(word) for word in set(words))
        if most / len(words) > 0.5:
            return True
    return False


class VisionClient:
    """OpenRouter vision over chat completions, with no SDK dependency."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        ledger: UsageLedger | None = None,
        *,
        allow_external: bool = True,
        max_tokens: int = 4096,
        retries: int = 3,
        backoff: float = 1.5,
        tile_depth: int = 2,
        tile_overlap: float = 0.08,
        provider_data_collection_deny: bool = True,
        provider_zero_retention: bool = False,
    ) -> None:
        key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        if not key:
            raise VisionError(
                "OPENROUTER_API_KEY is empty. Set it in .env before "
                "using vision, or call parse() without a VisionClient to get "
                "text-only parsing with scanned pages flagged."
            )
        self.api_key = key
        self.model = model or os.getenv("VISION_MODEL", DEFAULT_MODEL)
        self.base_url = (
            base_url or os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)
        ).rstrip("/")
        self.timeout = timeout
        self.ledger = ledger or UsageLedger()
        self.allow_external = allow_external
        self.max_tokens = max_tokens
        self.retries = max(1, retries)
        self.backoff = max(0.0, backoff)
        self.tile_depth = max(0, tile_depth)
        self.tile_overlap = tile_overlap
        self.provider_data_collection_deny = provider_data_collection_deny
        self.provider_zero_retention = provider_zero_retention
        self._http: Any = None
        self._last_result: VisionResult | None = None

    @property
    def calls(self) -> list[UsageRecord]:
        return [
            record
            for record in self.ledger.records
            if record.operation.startswith("vision.")
        ]

    @property
    def last_result(self) -> VisionResult | None:
        return self._last_result

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None

    def __enter__(self) -> "VisionClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def ocr(
        self,
        image: bytes,
        *,
        metadata: dict[str, Any] | None = None,
        prompt_type: str = "default",
        return_meta: bool = False,
    ) -> str | VisionResult:
        prompt = OCR_PROMPTS.get(prompt_type, OCR_PROMPTS["default"])
        return self._run(image, prompt, "ocr", metadata, prompt_type, return_meta)

    def describe(
        self,
        image: bytes,
        *,
        metadata: dict[str, Any] | None = None,
        prompt_type: str = "figure",
        return_meta: bool = False,
    ) -> str | VisionResult:
        prompt = FIGURE_PROMPTS.get(prompt_type, FIGURE_PROMPTS["figure"])
        return self._run(image, prompt, "figure", metadata, prompt_type, return_meta)

    def _run(
        self,
        image: bytes,
        prompt: str,
        kind: str,
        metadata: dict[str, Any] | None,
        prompt_type: str,
        return_meta: bool,
    ) -> str | VisionResult:
        if not self.allow_external:
            raise VisionError(
                "External vision is disabled. Set allow_external=True or "
                "ALLOW_EXTERNAL_VISION=true before sending page images."
            )
        result = self._generate(image, prompt, kind, metadata or {}, prompt_type, 0)
        self._last_result = result
        return result if return_meta else result.text

    def _client(self) -> Any:
        if self._http is None:
            import httpx

            self._http = httpx.Client(
                timeout=self.timeout, limits=httpx.Limits(max_connections=8)
            )
        return self._http

    def _payload(self, image: bytes, prompt: str) -> dict[str, Any]:
        data_url = "data:image/png;base64," + base64.b64encode(image).decode("ascii")
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        }
        provider: dict[str, Any] = {}
        if self.provider_data_collection_deny:
            provider["data_collection"] = "deny"
        if self.provider_zero_retention:
            provider["zdr"] = True
        if provider:
            payload["provider"] = provider
        return payload

    def _generate(
        self,
        image: bytes,
        prompt: str,
        kind: str,
        metadata: dict[str, Any],
        prompt_type: str,
        depth: int,
    ) -> VisionResult:
        text, finish_reason, usage = self._post(image, prompt, kind, metadata, depth)

        warnings: list[str] = []
        tiles = 1

        # Truncation. Re-run as overlapping strips, then stitch.
        if finish_reason == "length" and depth < self.tile_depth:
            tiles, text, tile_warnings = self._tiled(image, prompt, kind, metadata, depth)
            warnings.append(f"response hit max_tokens; retried as {tiles} tiles")
            warnings.extend(tile_warnings)
        elif finish_reason == "length":
            warnings.append(
                "response hit max_tokens at maximum tiling depth; transcription may be incomplete"
            )

        # Empty or implausible OCR output for an image that clearly has ink.
        ink = _ink_density(image) if kind == "ocr" else 0.0
        if kind == "ocr" and not text.strip() and ink > 0.005 and depth < self.tile_depth:
            tiles, text, tile_warnings = self._tiled(image, prompt, kind, metadata, depth)
            warnings.extend(["empty OCR response on an inked image", *tile_warnings])
        elif kind == "ocr" and not text.strip() and ink > 0.005:
            warnings.append("empty OCR response on an inked image at maximum tiling depth")
        elif kind == "ocr" and ink > 0.02 and len(text.split()) < 3:
            warnings.append("implausibly few words for the amount of ink")

        if _looks_like_loop(text):
            if depth < self.tile_depth:
                tiles, text, tile_warnings = self._tiled(image, prompt, kind, metadata, depth)
                warnings.extend(["degenerate repetition", *tile_warnings])
            if _looks_like_loop(text):
                warnings.append("degenerate repetition persisted after tiling")

        if kind == "ocr" and _REFUSAL.match(text):
            if depth < self.tile_depth:
                tiles, text, tile_warnings = self._tiled(image, prompt, kind, metadata, depth)
                warnings.extend(["model returned a refusal instead of a transcription", *tile_warnings])
            if _REFUSAL.match(text):
                warnings.append("model refused to transcribe")
                text = ""

        if finish_reason not in {"", "stop"} and finish_reason != "length":
            warnings.append(f"unexpected finish_reason {finish_reason}")

        return VisionResult(
            text=text.strip(),
            finish_reason=finish_reason,
            tiles=tiles,
            model=self.model,
            warnings=tuple(dict.fromkeys(warnings)),
            usage=usage,
        )

    def _tiled(
        self,
        image: bytes,
        prompt: str,
        kind: str,
        metadata: dict[str, Any],
        depth: int,
    ) -> tuple[int, str, list[str]]:
        try:
            tiles = _tile_strips(image, self.tile_overlap)
        except Exception as exc:
            return 1, "", [f"could not tile image: {type(exc).__name__}"]

        texts: list[str] = []
        warnings: list[str] = []
        finish_reasons: list[str] = []
        for index, tile in enumerate(tiles):
            tile_meta = {
                **metadata,
                "tile_index": index,
                "tile_depth": depth + 1,
            }
            text, finish_reason, _ = self._post(
                tile, prompt, kind, tile_meta, depth + 1
            )
            texts.append(text)
            finish_reasons.append(finish_reason)
            if finish_reason == "length":
                if depth + 1 < self.tile_depth:
                    nested_count, nested_text, nested_warnings = self._tiled(
                        tile, prompt, kind, metadata, depth + 1
                    )
                    texts[-1] = nested_text
                    warnings.extend(nested_warnings)
                else:
                    warnings.append(
                        f"tile {index} hit max_tokens at maximum tiling depth"
                    )

        if any(reason == "length" for reason in finish_reasons):
            warnings.append("one or more tiles were truncated")
        stitched = _stitch(texts)
        return len(tiles), stitched, warnings

    def _post(
        self,
        image: bytes,
        prompt: str,
        kind: str,
        metadata: dict[str, Any],
        depth: int,
    ) -> tuple[str, str, dict[str, Any]]:
        payload = self._payload(image, prompt)
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "luvparse",
        }

        last_error: Exception | None = None
        for attempt in range(self.retries):
            started = time.monotonic()
            try:
                response = self._client().post(
                    url, json=payload, headers=headers, timeout=self.timeout
                )
                if response.status_code in _RETRY_STATUS:
                    delay = self._retry_delay(response, attempt)
                    last_error = VisionError(
                        f"OpenRouter returned {response.status_code}"
                    )
                    if attempt < self.retries - 1:
                        time.sleep(delay)
                        continue
                    response.raise_for_status()

                if (
                    response.status_code == 404
                    and "data policy" in response.text.lower()
                ):
                    raise VisionError(
                        "Zero data retention is not available for this model. "
                        "Disable provider_zero_retention or choose a model with "
                        "a zero-retention endpoint. No image was accepted."
                    )

                response.raise_for_status()
                data = response.json()
                text = _completion_text(data)
                usage = dict(data.get("usage") or {})
                finish_reason = str(
                    (data.get("choices") or [{}])[0].get("finish_reason") or ""
                )
                reserved = {
                    "document_id",
                    "stage",
                    "operation",
                    "provider",
                    "model",
                    "seconds",
                    "usage",
                    "cost_usd",
                }
                self.ledger.record(
                    operation=f"vision.{kind}",
                    provider="openrouter",
                    model=self.model,
                    seconds=time.monotonic() - started,
                    usage=usage,
                    **{
                        key: value
                        for key, value in metadata.items()
                        if key not in reserved
                    },
                    document_id=metadata.get("document_id"),
                    stage=metadata.get("stage"),
                )
                return text, finish_reason, usage
            except VisionError:
                raise
            except Exception as exc:  # network, JSON shape, provider error
                last_error = exc
                if attempt < self.retries - 1:
                    time.sleep(self._backoff(attempt))
                    continue
                raise VisionError(
                    f"OpenRouter {kind} call failed after {self.retries} attempts: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

        raise VisionError(
            f"OpenRouter {kind} call failed: {type(last_error).__name__}"
        )

    def _retry_delay(self, response: Any, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
        return self._backoff(attempt)

    def _backoff(self, attempt: int) -> float:
        base = self.backoff * (2**attempt)
        return base + random.uniform(0, base * 0.25)


def _completion_text(data: dict[str, Any]) -> str:
    """Pull assistant text out of a chat completion, accepting both shapes."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise VisionError("Unexpected OpenRouter response shape") from exc
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    return ""


def ocr_page(
    image: bytes,
    *,
    client: VisionClient,
    metadata: dict[str, Any] | None = None,
    prompt_type: str = "default",
    return_meta: bool = False,
) -> str | VisionResult:
    return client.ocr(
        image, metadata=metadata, prompt_type=prompt_type, return_meta=return_meta
    )


def describe_figure(
    image: bytes,
    *,
    client: VisionClient,
    metadata: dict[str, Any] | None = None,
    prompt_type: str = "figure",
    return_meta: bool = False,
) -> str | VisionResult:
    return client.describe(
        image, metadata=metadata, prompt_type=prompt_type, return_meta=return_meta
    )


def vision_from_env(
    ledger: UsageLedger | None = None,
    *,
    allow_external: bool | None = None,
    model: str | None = None,
    config: Any = None,
) -> VisionClient | None:
    """A client when a key is set and external calls are allowed.

    ``allow_external=None`` preserves the old behaviour for callers that passed
    a client explicitly. New pipeline callers pass the config value.
    """
    if not os.getenv("OPENROUTER_API_KEY"):
        return None
    if allow_external is False:
        return None
    kwargs: dict[str, Any] = {"ledger": ledger, "model": model}
    if allow_external is not None:
        kwargs["allow_external"] = allow_external
    if config is not None:
        kwargs.update(
            allow_external=config.allow_external,
            max_tokens=config.vision_max_tokens,
            retries=config.vision_retries,
            timeout=config.vision_timeout,
            backoff=config.vision_backoff,
            tile_depth=config.tile_max_depth,
            tile_overlap=config.tile_overlap,
            provider_data_collection_deny=config.provider_data_collection_deny,
            provider_zero_retention=config.provider_zero_retention,
        )
        kwargs["model"] = config.vision_model or model
    return VisionClient(**kwargs)

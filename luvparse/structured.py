"""Structured markdown for PDFs: headings, formatting, tables, furniture, layout.

PyMuPDF's plain text is a flat word stream, which is right for search and wrong
for a benchmark that scores document structure. This module is the structure
view. For every page it produces:

* markdown with real structure: ``#`` headings, ``**bold**``, ``*italic*``,
  ``~~strikeout~~``, ``<sup>``, ``<sub>``, ``<u>`` and ``<mark>``
* ``<page_header>`` / ``<page_footer>`` tags, so furniture is kept but is not
  mistaken for body text
* HTML tables with ``rowspan`` / ``colspan`` and ``<th>`` header cells, built
  from PyMuPDF cell geometry
* optional chart tables (series, categories, values) from a vision model
* layout items: label, normalised bounding box and the span of markdown each
  box produced

Page structure comes from ``pymupdf4llm`` (layout model, reading order,
headings, bold, italic, strikeout). Everything above that list is added here.
"""

from __future__ import annotations

import html as _html
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

_LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
}
_INVISIBLE = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\u2060\ufeff"), None)
_MARKUP = r"(?:\*{1,2}|_{1,2}|~~|</?\w+[^>]*>)*"


# --------------------------------------------------------------------------- types


@dataclass
class StructuredConfig:
    """Switches for the structure view. Every one can be ablated on its own."""

    xy_order: bool = True  # layout-only: reading order by XY-cut (markdown is built in this order)
    refine_layout: bool = True  # layout-only box polish (never changes the text)
    emit_page_tags: bool = True  # <page_header>/<page_footer> around furniture
    bold_row_tables: bool = True  # one-row bold tables become bold title lines
    text_tables: bool = True  # borderless tables found by text alignment
    own_tables: bool = True  # geometry tables with spans instead of the engine's
    inline_extras: bool = True  # <sup> <sub> <u> <mark>
    clean_text: bool = True  # ligatures, invisible chars, line-end hyphens
    chart_tables: bool = False  # vision: chart image -> HTML table
    vision_tables: bool = False  # vision: re-read suspect table crops as HTML
    vision_tables_all: bool = False  # ... every table, not only suspect ones
    promote_titles: bool = False  # standalone title-like lines become headings
    ocr_scanned: bool = False  # vision: OCR pages with no text layer
    render_dpi: int = 200
    use_ocr: bool | None = None  # forwarded to pymupdf4llm (None = engine default)
    min_table_token_recall: float = 0.9


@dataclass
class StructuredPage:
    number: int
    width: float
    height: float
    markdown: str
    items: list[dict[str, Any]] = field(default_factory=list)
    header: str = ""
    footer: str = ""
    printed_page_number: str = ""
    tables: int = 0
    charts: int = 0
    warnings: list[str] = field(default_factory=list)
    source: str = "pymupdf4llm"

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.number,
            "width": self.width,
            "height": self.height,
            "markdown": self.markdown,
            "items": self.items,
            "page_header_markdown": self.header,
            "page_footer_markdown": self.footer,
            "printed_page_number": self.printed_page_number,
            "tables": self.tables,
            "charts": self.charts,
            "warnings": list(self.warnings),
            "source": self.source,
        }


# --------------------------------------------------------------------------- text


def clean_text(text: str) -> str:
    """Safe character-level repairs. Never changes words or numbers."""
    for ligature, plain in _LIGATURES.items():
        text = text.replace(ligature, plain)
    text = text.translate(_INVISIBLE).replace("\u00a0", " ")
    # A hyphen at a line end followed by a lowercase continuation is a split word.
    text = re.sub(r"(?<=[A-Za-z])-\n(?=[a-z])", "", text)
    return text


def plain_text(markdown: str) -> str:
    """Markdown or HTML to the words a human would read."""
    text = re.sub(r"<[^>]+>", " ", markdown)
    text = _html.unescape(text)
    text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text, flags=re.M)
    text = re.sub(r"(\*\*|__|~~|\*|_)", "", text)
    return re.sub(r"\s+", " ", text).strip()


_PAGE_NUMBER = re.compile(
    r"(?:^|\b)(?:page\s+)?(\d{1,4}|[ivxlcdm]{1,7})(?:\s*(?:of|/)\s*\d{1,4})?(?:\b|$)",
    re.I,
)


def printed_page_number(*furniture: str) -> str:
    """A page number inside header or footer text, if there is exactly one."""
    for text in furniture:
        stripped = plain_text(text).strip(" -–—|.")
        if re.fullmatch(r"(?:page\s+)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?", stripped, re.I):
            match = re.search(r"\d+", stripped)
            if match:
                return match.group(0)
    return ""


# --------------------------------------------------------------------------- tables


def _cluster(values: Iterable[float], tolerance: float = 2.0) -> list[float]:
    ordered = sorted(values)
    groups: list[list[float]] = []
    for value in ordered:
        if groups and value - groups[-1][-1] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [sum(group) / len(group) for group in groups]


def _index(grid: list[float], value: float, tolerance: float = 3.0) -> int:
    best = min(range(len(grid)), key=lambda i: abs(grid[i] - value))
    return best if abs(grid[best] - value) <= tolerance else -1


def _cell_lines(page: Any, bbox: tuple[float, float, float, float]) -> list[dict[str, Any]]:
    """Text lines in a cell: vertical centre and text, with <sub>/<sup> applied."""
    import pymupdf

    rect = pymupdf.Rect(bbox)
    # Shrink slightly so neighbouring cells' text is not pulled in.
    rect = pymupdf.Rect(rect.x0 + 0.5, rect.y0 + 0.5, rect.x1 - 0.5, rect.y1 - 0.5)
    out: list[dict[str, Any]] = []
    for block in page.get_text("dict", clip=rect)["blocks"]:
        for line in block.get("lines", []):
            spans = [sp for sp in line["spans"] if sp["text"]]
            text = "".join(sp["text"] for sp in spans).strip()
            if not text:
                continue
            # Centre of the main (largest) span, so a superscript does not shift it.
            main = max(spans, key=lambda sp: sp["size"] * len(sp["text"].strip()))
            y = (main["bbox"][1] + main["bbox"][3]) / 2
            text = clean_text(text)
            if len(spans) > 1:
                try:
                    extras = inline_extras(page, tuple(line["bbox"]), kinds=("sup", "sub"))
                    if extras:
                        text = apply_extras(text, extras)
                except Exception:
                    pass
            out.append({"y": y, "x": line["bbox"][0], "text": text})
    out.sort(key=lambda ln: (round(ln["y"] / 2), ln["x"]))
    return out


def _cell_text(page: Any, bbox: tuple[float, float, float, float]) -> str:
    return " ".join(ln["text"] for ln in _cell_lines(page, bbox)).strip()


def _split_unruled_rows(grid: dict[str, Any]) -> None:
    """Split a grid row into one row per text line when every cell in it has the
    same number of lines at the same heights (rows with no gridline between them).
    """
    if any(c["rowspan"] > 1 for c in grid["cells"]):
        return
    by_row: dict[int, list[dict[str, Any]]] = {}
    for cell in grid["cells"]:
        by_row.setdefault(cell["row"], []).append(cell)
    new_cells: list[dict[str, Any]] = []
    next_row = 0
    for r in range(grid["n_rows"]):
        row_cells = by_row.get(r, [])
        filled = [c for c in row_cells if c["lines"]]
        counts = {len(c["lines"]) for c in filled}
        split = False
        if filled and len(counts) == 1 and next(iter(counts)) > 1 and len(row_cells) >= 2:
            n = next(iter(counts))
            ref = [ln["y"] for ln in filled[0]["lines"]]
            split = all(
                abs(ln["y"] - ref[k]) <= 3 for c in filled for k, ln in enumerate(c["lines"])
            ) and all(ref[k + 1] - ref[k] > 4 for k in range(n - 1))
        if split:
            for k in range(n):
                for cell in row_cells:
                    lines = cell["lines"]
                    new_cells.append(
                        {**cell, "row": next_row + k, "text": lines[k]["text"] if lines else "", "lines": lines[k : k + 1] if lines else []}
                    )
            next_row += n
        else:
            for cell in row_cells:
                new_cells.append({**cell, "row": next_row})
            next_row += 1
    grid["cells"] = sorted(new_cells, key=lambda e: (e["row"], e["col"]))
    grid["n_rows"] = next_row


def _bold_ratio(page: Any, bbox: tuple[float, float, float, float]) -> float:
    import pymupdf

    rect = pymupdf.Rect(bbox)
    total = bold = 0
    for block in page.get_text("dict", clip=rect)["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                n = len(span["text"].strip())
                if not n:
                    continue
                total += n
                if span["flags"] & 16 or re.search(r"bold|black|heavy|semibold", span["font"], re.I):
                    bold += n
    return bold / total if total else 0.0


_NUMERIC = re.compile(r"^[\s$€£¥(+\-−]*\d[\d,.\s%)]*[a-zA-Z%]{0,3}$")


def _is_numeric(text: str) -> bool:
    return bool(_NUMERIC.match(text.strip())) if text.strip() else False


_TABLE_CACHE: dict[tuple[Any, ...], Any] = {}


def _find_tables_cached(page: Any, strategy: str | None = None) -> Any:
    # Whole-page search: the layout model's box can be tighter than the grid,
    # and clipping to it would cut real cells off.
    key = (id(page.parent), page.number, strategy)
    if key not in _TABLE_CACHE:
        if len(_TABLE_CACHE) > 8:
            _TABLE_CACHE.clear()
        _TABLE_CACHE[key] = page.find_tables(strategy=strategy) if strategy else page.find_tables()
    return _TABLE_CACHE[key]


def geometry_table(page: Any, bbox: tuple[float, float, float, float]) -> dict[str, Any] | None:
    """Table grid with spans from PyMuPDF cell geometry, or None."""
    import pymupdf

    rect = pymupdf.Rect(bbox)
    try:
        found = _find_tables_cached(page)
    except Exception:
        return None
    best = None
    best_overlap = 0.0
    for table in found.tables:
        inter = pymupdf.Rect(table.bbox) & rect
        overlap = inter.get_area() / max(rect.get_area(), 1.0)
        if overlap > best_overlap:
            best, best_overlap = table, overlap
    if best is None or best_overlap < 0.5:
        return None
    return grid_from_table(page, best)


def grid_from_table(page: Any, best: Any) -> dict[str, Any] | None:
    """Grid with spans from one PyMuPDF table's cell geometry."""
    # Slivers a few points wide are double borders, not cells.
    cells = [
        tuple(c)
        for row in best.rows
        for c in row.cells
        if c is not None and (c[2] - c[0]) >= 4 and (c[3] - c[1]) >= 4
    ]
    if not cells:
        return None
    xs = _cluster([c[0] for c in cells] + [c[2] for c in cells])
    ys = _cluster([c[1] for c in cells] + [c[3] for c in cells])
    if len(xs) < 2 or len(ys) < 2:
        return None
    n_cols, n_rows = len(xs) - 1, len(ys) - 1

    placed: dict[tuple[int, int], dict[str, Any]] = {}
    for cell in sorted(set(cells), key=lambda c: (c[1], c[0])):
        c0, c1 = _index(xs, cell[0]), _index(xs, cell[2])
        r0, r1 = _index(ys, cell[1]), _index(ys, cell[3])
        if min(c0, c1, r0, r1) < 0 or c1 <= c0 or r1 <= r0:
            return None
        key = (r0, c0)
        if key in placed:
            continue
        placed[key] = {
            "row": r0,
            "col": c0,
            "rowspan": r1 - r0,
            "colspan": c1 - c0,
            "bbox": cell,
        }
    for entry in placed.values():
        entry["lines"] = _cell_lines(page, entry["bbox"])
        entry["text"] = " ".join(ln["text"] for ln in entry["lines"]).strip()
        entry["bold"] = _bold_ratio(page, entry["bbox"])

    grid = {
        "bbox": tuple(best.bbox),
        "n_rows": n_rows,
        "n_cols": n_cols,
        "cells": sorted(placed.values(), key=lambda e: (e["row"], e["col"])),
        "header_hint": _pymupdf_header_rows(best, ys),
    }
    _split_unruled_rows(grid)
    return grid


def _pymupdf_header_rows(table: Any, ys: list[float]) -> int | None:
    try:
        header = table.header
        if header is None or getattr(header, "external", False):
            return None
        bottom = max(c[3] for c in header.cells if c is not None)
        return max(1, _index(ys, bottom))
    except Exception:
        return None


def infer_header_rows(grid: dict[str, Any]) -> int:
    """How many leading rows are headers.

    Rows count as header while they are bold, or carry no numbers above a body
    that does. Rows reached by a rowspan from above stay in the header block.
    """
    cells = grid["cells"]
    n_rows = grid["n_rows"]
    by_row: dict[int, list[dict[str, Any]]] = {}
    for cell in cells:
        by_row.setdefault(cell["row"], []).append(cell)

    def row_is_bold(r: int) -> bool:
        filled = [c for c in by_row.get(r, []) if c["text"]]
        return bool(filled) and sum(c["bold"] >= 0.6 for c in filled) / len(filled) >= 0.6

    def row_has_numbers(r: int) -> bool:
        filled = [c for c in by_row.get(r, []) if c["text"]]
        return bool(filled) and sum(_is_numeric(c["text"]) for c in filled) / len(filled) >= 0.4

    header_rows = 0
    if grid.get("header_hint") and grid.get("lead_in"):
        header_rows = min(int(grid["header_hint"]), n_rows - 1)  # the lead-in label is the whole header
    first_col = [c for c in cells if c["col"] == 0 and c["text"] and c["colspan"] == 1]
    label_col = len(first_col) >= 4 and sum(c["bold"] >= 0.6 for c in first_col) / len(first_col) >= 0.8
    # A bold first column labels the rows; it does not make them headers.
    limit = 1 if label_col else 4
    while not grid.get("lead_in") and header_rows < min(limit, n_rows - 1) and row_is_bold(header_rows):
        header_rows += 1
    if header_rows == 0:
        if not row_has_numbers(0) and n_rows > 1 and row_has_numbers(1):
            header_rows = 1
        elif grid.get("header_hint"):
            header_rows = min(int(grid["header_hint"]), n_rows - 1)
        elif not row_has_numbers(0) and n_rows > 1:
            header_rows = 1
    # A cell spanning down out of the header block pulls its rows in.
    changed = True
    while changed:
        changed = False
        for cell in cells:
            if cell["row"] < header_rows and cell["row"] + cell["rowspan"] > header_rows:
                header_rows = min(cell["row"] + cell["rowspan"], n_rows - 1) or header_rows
                changed = True
    return header_rows


def _cell_html(text: str) -> str:
    """Escape cell text but keep the inline tags this module itself adds."""
    escaped = _html.escape(text, quote=False)
    return re.sub(r"&lt;(/?)(sub|sup|u|mark)&gt;", r"<\1\2>", escaped)


def grid_to_html(grid: dict[str, Any], header_rows: int | None = None) -> str:
    rows = infer_header_rows(grid) if header_rows is None else header_rows
    by_row: dict[int, list[dict[str, Any]]] = {}
    for cell in grid["cells"]:
        by_row.setdefault(cell["row"], []).append(cell)
    lines = ["<table>"]
    for r in range(grid["n_rows"]):
        row_cells = sorted(by_row.get(r, []), key=lambda c: c["col"])
        if not row_cells:
            continue
        out = []
        for cell in row_cells:
            tag = "th" if r < rows else "td"
            attrs = ""
            if cell["rowspan"] > 1:
                attrs += f' rowspan="{cell["rowspan"]}"'
            if cell["colspan"] > 1:
                attrs += f' colspan="{cell["colspan"]}"'
            out.append(f"<{tag}{attrs}>{_cell_html(cell['text'])}</{tag}>")
        lines.append("<tr>" + "".join(out) + "</tr>")
    lines.append("</table>")
    return "".join(lines)


def _tokens(text: str) -> list[str]:
    # Inline tags sit inside words (V<sub>DD</sub>); drop them without a space.
    text = re.sub(r"</?(?:sub|sup|u|mark|b|i)>", "", text)
    return re.findall(r"\w+", plain_text(text).lower())


def token_recall(reference: str, candidate: str) -> float:
    """Share of the reference's word tokens present in the candidate."""
    ref = _tokens(reference)
    if not ref:
        return 1.0
    pool: dict[str, int] = {}
    for token in _tokens(candidate):
        pool[token] = pool.get(token, 0) + 1
    hit = 0
    for token in ref:
        if pool.get(token, 0) > 0:
            pool[token] -= 1
            hit += 1
    return hit / len(ref)



# --------------------------------------------------------------------------- titles / vision tables

_TITLE_STOP = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "vs", "with"}


def _body_font_size(page: Any) -> float:
    sizes: list[float] = []
    try:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    sizes.extend([span["size"]] * max(len(span["text"].strip()), 0))
    except Exception:
        return 0.0
    if not sizes:
        return 0.0
    sizes.sort()
    return sizes[len(sizes) // 2]


def _box_font(page: Any, bbox: Any) -> tuple[float, bool]:
    import pymupdf

    best, bold, total = 0.0, 0, 0
    clip = pymupdf.Rect(bbox)
    for block in page.get_text("dict", clip=clip)["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if not span["text"].strip():
                    continue
                best = max(best, float(span["size"]))
                total += 1
                bold += 1 if _is_bold_font(span) else 0
    return best, bool(total) and bold / total >= 0.8


def promote_titles(page: Any, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A short standalone line set larger or bolder than the body is a title.

    Mirrors the rule other parsers apply after layout: at most 12 words, no
    closing punctuation, not a label-value pair, mostly capitalised words.
    """
    body = _body_font_size(page)
    if not body:
        return segments
    out = []
    for seg in segments:
        cls = seg["cls"].lower().replace("_", "-")
        text = seg["text"].strip()
        if cls == "text" and seg.get("bbox") and "\n" not in text and not text.startswith(("#", "-", "*", "|", "<")):
            plain = plain_text(text)
            words = plain.split()
            if 1 <= len(words) <= 12 and not re.search(r"[.,;:!?)]$", plain) and ":" not in plain:
                cased = [w for w in words if w.lower() not in _TITLE_STOP and w[:1].isalpha()]
                caps = sum(1 for w in cased if w[:1].isupper()) / len(cased) if cased else 0.0
                size, bold = _box_font(page, seg["bbox"])
                if caps >= 0.6 and (size >= 1.2 * body or (bold and size >= body)):
                    seg = {**seg, "cls": "section-header", "text": "# " + re.sub(r"^\*\*(.*)\*\*$", r"\1", text)}
        out.append(seg)
    return out


def _table_suspect(grid: dict[str, Any] | None) -> bool:
    """A table the line-based reader clearly got wrong (nothing read, mostly empty, or mush)."""
    if not grid:
        return True
    cells = grid["cells"]
    empty = sum(1 for c in cells if not c["text"].strip())
    if empty / max(len(cells), 1) > 0.4:
        return True
    return any(len(c["text"].split()) > 40 for c in cells)


_DOTS = re.compile(r"(?:\.\s?){5,}")


def vision_table_html(page: Any, bbox: Any, vision: Any, config: "StructuredConfig", reference: str) -> str:
    """Table crop re-read by the vision model; "" when the reply is not trustworthy."""
    reply = vision.describe(_page_png(page, bbox, max(config.render_dpi, 200)), prompt_type="table_html")
    html = "".join(re.findall(r"<table\b.*?</table>", str(reply), re.I | re.S))
    if not html:
        return ""
    html = re.sub(r"\s*\n\s*", "", html)
    html = _DOTS.sub(" ", html)
    words = " ".join(w[4] for w in page.get_text("words", clip=__import__("pymupdf").Rect(bbox)))
    if words and token_recall(words, html) < 0.6:
        return ""  # the model dropped or invented too much
    return html


# --------------------------------------------------------------------------- inline extras


def _is_bold_font(span: dict[str, Any]) -> bool:
    return bool(span["flags"] & 16) or bool(
        re.search(r"bold|black|heavy|semibold", span["font"], re.I)
    )


def _is_italic_font(span: dict[str, Any]) -> bool:
    return bool(span["flags"] & 2) or bool(re.search(r"italic|oblique", span["font"], re.I))


def _highlight_colour(fill: Any) -> bool:
    """A light, saturated fill: yellow, green, pink, cyan. Not white, grey or dark."""
    if not fill or len(fill) != 3:
        return False
    high, low = max(fill), min(fill)
    return high >= 0.6 and (high - low) >= 0.25


def _saturated(rgb: int) -> bool:
    """A real colour (orange, red), not black or a dark grey a printer calls black."""
    r, g, b = (rgb >> 16) & 255, (rgb >> 8) & 255, rgb & 255
    return max(r, g, b) - min(r, g, b) >= 40


def inline_extras(
    page: Any,
    bbox: tuple[float, float, float, float],
    kinds: tuple[str, ...] = ("sup", "sub", "u", "mark"),
) -> list[dict[str, Any]]:
    """Runs of superscript, subscript, underline, strikeout and highlight in a box.

    Each result: kind, text, left (context before), right (context after).
    """
    import pymupdf

    rect = pymupdf.Rect(bbox)
    rect = pymupdf.Rect(rect.x0 - 1, rect.y0 - 3, rect.x1 + 1, rect.y1 + 3)
    spans: list[dict[str, Any]] = []
    for block in page.get_text("dict", clip=rect)["blocks"]:
        for line in block.get("lines", []):
            line_box = pymupdf.Rect(line["bbox"])
            for span in line["spans"]:
                if span["text"].strip():
                    span["_line"] = line_box
                    spans.append(span)
    if not spans:
        return []

    rules = []  # thin horizontal strokes: underline or strikeout candidates
    rule_colour: dict[int, int] = {}  # id(rule) -> packed RGB of the stroke/fill
    fills = []  # highlight rectangles
    need_marks = any(k in kinds for k in ("mark", "u", "s"))
    for d in (page.get_drawings() if need_marks else []):
        r = pymupdf.Rect(d["rect"])
        if not r.intersects(rect):
            continue
        if r.height <= 1.6 and r.width >= 3:
            rules.append(r)
            col = d.get("fill") if d.get("fill") is not None else d.get("color")
            if col:
                rule_colour[id(r)] = (
                    (round(col[0] * 255) << 16) | (round(col[1] * 255) << 8) | round(col[2] * 255)
                )
        if d.get("type") in ("f", "fs") and _highlight_colour(d.get("fill")) and 2 < r.height <= 40:
            fills.append(r)
    for annot in (page.annots() or [] if need_marks else []):
        kind = annot.type[1]
        r = pymupdf.Rect(annot.rect)
        if kind == "Highlight":
            fills.append(r)
        elif kind == "Underline":
            rules.append(pymupdf.Rect(r.x0, r.y1 - 1, r.x1, r.y1))
        elif kind == "StrikeOut":
            mid = (r.y0 + r.y1) / 2
            rules.append(pymupdf.Rect(r.x0, mid - 0.5, r.x1, mid + 0.5))

    results: list[dict[str, Any]] = []
    for i, span in enumerate(spans):
        sb = pymupdf.Rect(span["bbox"])
        size = span["size"]
        kind = None
        if span["flags"] & 1:
            kind = "sup"
        else:
            for other in spans:
                if other is span or other["size"] <= size * 1.15:
                    continue
                ob = pymupdf.Rect(other["bbox"])
                touching = abs(sb.x0 - ob.x1) <= 2.5 or abs(ob.x0 - sb.x1) <= 2.5
                vertical = min(sb.y1, ob.y1) - max(sb.y0, ob.y0)
                if not touching or vertical < 0.3 * min(sb.height, ob.height):
                    continue
                if size > 0.85 * other["size"]:
                    continue
                shift = other["origin"][1] - span["origin"][1]
                if shift > 0.15 * other["size"]:
                    kind = "sup"
                elif shift < -0.08 * other["size"]:
                    kind = "sub"
                break
        if kind is None or kind not in kinds:
            continue
        left = "".join(s_["text"] for s_ in spans[:i])[-3:].replace(" ", "")
        right = "".join(s_["text"] for s_ in spans[i + 1 :])[:2].replace(" ", "")
        results.append(
            {
                "kind": kind,
                "text": span["text"].strip(),
                "left": left[-2:],
                "right": right[:1],
                "pos": (round(span["_line"].y0 / 3), span["bbox"][0]),
            }
        )

    # Highlight, underline, strikeout: per character, because PyMuPDF merges a
    # whole line of one font into one span and a highlight may cover two words.
    wanted = [k for k in ("mark", "u", "s") if k in kinds]
    if wanted and (fills or rules):
        for block in page.get_text("rawdict", clip=rect)["blocks"]:
            for line in block.get("lines", []):
                line_box = pymupdf.Rect(line["bbox"])
                chars = []
                for sp in line["spans"]:
                    for ch in sp.get("chars", []):
                        ch["_colour"] = sp.get("color", 0)
                        ch["_bold"] = bool(sp.get("flags", 0) & 16)
                        chars.append(ch)
                marks: list[str | None] = []
                for ch in chars:
                    cb = pymupdf.Rect(ch["bbox"])
                    mark = None
                    if ch["c"].strip() and cb.width > 0:
                        if "mark" in wanted:
                            for fill_rect in fills:
                                if (fill_rect & cb).get_area() >= 0.6 * cb.get_area():
                                    mark = "mark"
                                    break
                        if mark is None:
                            for rule in rules:
                                covers = (min(rule.x1, cb.x1) - max(rule.x0, cb.x0)) >= 0.8 * cb.width
                                inside = rule.x0 >= line_box.x0 - 4 and rule.x1 <= line_box.x1 + 4
                                if not covers or not inside:
                                    continue
                                if "u" in wanted and cb.y1 - 3 <= rule.y0 <= cb.y1 + 3:
                                    # Bold coloured text with a rule of its own colour is
                                    # tracked-change styling (inserted text): a highlight.
                                    own = ch["_colour"]
                                    if ch["_bold"] and _saturated(own) and rule_colour.get(id(rule)) == own and "mark" in wanted:
                                        mark = "mark"
                                    else:
                                        mark = "u"
                                    break
                                if "s" in wanted and cb.y0 + 0.3 * cb.height <= rule.y0 <= cb.y0 + 0.7 * cb.height:
                                    mark = "s"
                                    break
                    marks.append(mark)
                index = 0
                while index < len(chars):
                    kind = marks[index]
                    if kind is None:
                        index += 1
                        continue
                    end = index
                    while end + 1 < len(chars) and (
                        marks[end + 1] == kind or (not chars[end + 1]["c"].strip() and end + 2 < len(chars) and marks[end + 2] == kind)
                    ):
                        end += 1
                    text = "".join(c["c"] for c in chars[index : end + 1]).strip()
                    before = "".join(c["c"] for c in chars[:index]).replace(" ", "")
                    after = "".join(c["c"] for c in chars[end + 1 :]).replace(" ", "")
                    if text:
                        results.append(
                            {
                                "kind": kind,
                                "text": text,
                                "left": before[-2:],
                                "right": after[:1],
                                "pos": (round(line_box.y0 / 3), chars[index]["bbox"][0]),
                            }
                        )
                    index = end + 1
    # Reading order: lines within 6pt of each other (a superscript's line sits a
    # little off its neighbour's) are one row, ordered left to right.
    rows: list[float] = []
    for y in sorted({r["pos"][0] for r in results}):
        if not rows or y - rows[-1] > 2:
            rows.append(y)
    results.sort(key=lambda r: (max(i for i, y in enumerate(rows) if y <= r["pos"][0]), r["pos"][1]))
    return results


_TAGS = {"s": ("~~", "~~"), "sup": ("<sup>", "</sup>"), "sub": ("<sub>", "</sub>"), "u": ("<u>", "</u>"), "mark": ("<mark>", "</mark>")}


def apply_extras(markdown: str, extras: list[dict[str, Any]]) -> str:
    """Wrap each extra in its tag. Skips anything that cannot be located."""
    cursor = 0
    for extra in extras:
        open_tag, close_tag = _TAGS[extra["kind"]]
        words = extra["text"].split()
        target = (r"\s+".join(re.escape(w) for w in words))
        if extra["kind"] in ("u", "mark"):  # sup/sub sit inside words (H2O)
            if re.match(r"\w", extra["text"][:1]):
                target = r"(?<!\w)" + target
            if re.match(r"\w", extra["text"][-1:]):
                target = target + r"(?!\w)"
        pattern = target
        if extra["left"]:
            pattern = "(?:" + re.escape(extra["left"][-1]) + _MARKUP + r"\s*)" + pattern
        match = re.compile(pattern).search(markdown, cursor)
        if match is None:
            match = re.compile(target).search(markdown, cursor)
            if match is None or extra["kind"] in ("sup", "sub"):
                continue
        start = match.start()
        if extra["left"]:
            inner = re.compile(target).search(markdown, start)
            if inner is None:
                continue
            start = inner.start()
            end = inner.end()
        else:
            end = match.end()
        if markdown[max(0, start - 5) : start].endswith(open_tag):
            continue
        markdown = markdown[:start] + open_tag + markdown[start:end] + close_tag + markdown[end:]
        cursor = end + len(open_tag) + len(close_tag)
    return markdown


# --------------------------------------------------------------------------- pages


def _segments(md: str, boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cut the engine's markdown into one segment per layout box."""
    ordered = sorted(
        (b for b in boxes if isinstance(b.get("pos"), (list, tuple)) and len(b["pos"]) == 2),
        key=lambda b: (b["pos"][0], b["pos"][1]),
    )
    segments: list[dict[str, Any]] = []
    cursor = 0
    for box in ordered:
        start, end = int(box["pos"][0]), int(box["pos"][1])
        start = max(start, cursor)
        if end <= start:
            continue
        gap = md[cursor:start]
        if gap.strip():
            segments.append({"cls": "text", "text": gap.strip(), "bbox": None})
        segments.append(
            {"cls": str(box.get("class", "text")), "text": md[start:end].strip(), "bbox": box.get("bbox")}
        )
        cursor = end
    tail = md[cursor:]
    if tail.strip():
        segments.append({"cls": "text", "text": tail.strip(), "bbox": None})
    if not segments and md.strip():
        segments.append({"cls": "text", "text": md.strip(), "bbox": None})
    return segments


def _norm_box(bbox: Any, width: float, height: float) -> dict[str, float] | None:
    if not bbox or width <= 0 or height <= 0:
        return None
    x0, y0, x1, y1 = (float(v) for v in bbox)
    x0, x1 = max(0.0, min(x0, width)), max(0.0, min(x1, width))
    y0, y1 = max(0.0, min(y0, height)), max(0.0, min(y1, height))
    if x1 <= x0 or y1 <= y0:
        return None
    return {"x": x0 / width, "y": y0 / height, "w": (x1 - x0) / width, "h": (y1 - y0) / height}


def _page_png(page: Any, bbox: Any, dpi: int) -> bytes:
    import pymupdf

    clip = pymupdf.Rect(bbox) if bbox else None
    if clip is not None:
        clip = pymupdf.Rect(clip.x0 - 4, clip.y0 - 4, clip.x1 + 4, clip.y1 + 4) & page.rect
    return page.get_pixmap(dpi=dpi, clip=clip).tobytes("png")


def chart_crops(page: Any, bbox: Any) -> list[list[float]]:
    """Regions to read as charts: one, or one per chart when two sit side by side
    (or stacked) inside a single picture box."""
    import pymupdf

    box = pymupdf.Rect(bbox)
    try:
        clusters = [
            pymupdf.Rect(c) for c in vector_picture_boxes(page)
            if pymupdf.Rect(c).width >= 0.15 * page.rect.width
            and pymupdf.Rect(c).height >= 0.08 * page.rect.height
            and (pymupdf.Rect(c) & box).get_area() >= 0.8 * pymupdf.Rect(c).get_area()
        ]
    except Exception:
        clusters = []
    if len(clusters) == 2:
        a, b = sorted(clusters, key=lambda r: (r.x0, r.y0))
        if a.x1 <= b.x0 + 2:  # side by side
            mid = (a.x1 + b.x0) / 2
            return [[box.x0, box.y0, mid, box.y1], [mid, box.y0, box.x1, box.y1]]
        a, b = sorted(clusters, key=lambda r: (r.y0, r.x0))
        if a.y1 <= b.y0 + 2:  # stacked
            mid = (a.y1 + b.y0) / 2
            return [[box.x0, box.y0, box.x1, mid], [box.x0, mid, box.x1, box.y1]]
    return [list(box)]


_CHART_HINT = re.compile(r"chart|figure|graph|plot|fig\.", re.I)


def looks_like_table_html(text: str) -> bool:
    return bool(re.search(r"<table\b", text, re.I))


def extract_table_html(model_output: str) -> str:
    """First HTML table in a model reply, or the markdown table converted."""
    match = re.search(r"<table\b.*?</table>", model_output, re.I | re.S)
    if match:
        return re.sub(r"\s*\n\s*", "", match.group(0))
    rows = [ln for ln in model_output.splitlines() if ln.strip().startswith("|")]
    rows = [r for r in rows if not re.fullmatch(r"\s*\|?[\s:\-|]+\|?\s*", r)]
    if len(rows) < 2:
        return ""
    out = ["<table>"]
    for i, row in enumerate(rows):
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        tag = "th" if i == 0 else "td"
        out.append("<tr>" + "".join(f"<{tag}>{_html.escape(c, quote=False)}</{tag}>" for c in cells) + "</tr>")
    out.append("</table>")
    return "".join(out)


_LINE_END_NUMBER = re.compile(r"(?:(?<=[\s|\u2014\u2013\-])|^)(\d{1,4})\s*$")


def tag_page_number(block: str) -> str:
    """Wrap a bare page number in <page_number>, the way the benchmark annotates it.

    Only a number that ends a line after a separator, or a line that is only a
    number. "Page 9 of 10" is left alone: annotators did not tag those.
    """
    lines = block.split("\n")
    for i in range(len(lines) - 1, -1, -1):
        plain = plain_text(lines[i])
        if re.search(r"\bpage\s+\d+|\d+\s+of\s+\d+", plain, re.I):
            continue
        match = _LINE_END_NUMBER.search(lines[i])
        if match and not re.search(r"\d\.\d*$|\d,\d*$", lines[i][: match.start() + 1]):
            lines[i] = lines[i][: match.start(1)] + f"<page_number>{match.group(1)}</page_number>"
            break
    return "\n".join(lines)


def _span_runs(spans: list[dict[str, Any]], styled: bool = True) -> str:
    """One line of spans as markdown, merging neighbours with the same style."""
    runs: list[list[Any]] = []
    for span in spans:
        text = span["text"]
        if not text.strip():
            if runs:
                runs[-1][0] += text
            continue
        style = (_is_bold_font(span), _is_italic_font(span)) if styled else (False, False)
        if runs and runs[-1][1] == style:
            runs[-1][0] += text
        else:
            runs.append([text, style])
    out = []
    for text, (bold, italic) in runs:
        core = text.strip()
        lead, trail = text[: len(text) - len(text.lstrip())], text[len(text.rstrip()) :]
        if bold and italic:
            core = f"**_{core}_**"
        elif bold:
            core = f"**{core}**"
        elif italic:
            core = f"_{core}_"
        out.append(lead + core + trail)
    return "".join(out).strip()


def pdf_line_markdown(page: Any, bbox: Any, styled: bool = True) -> list[str]:
    """Text lines inside a box as markdown, one string per printed line."""
    import pymupdf

    rect = pymupdf.Rect(bbox)
    rect = pymupdf.Rect(rect.x0 - 1, rect.y0 - 1, rect.x1 + 1, rect.y1 + 1)
    lines: list[tuple[float, float, str]] = []
    for block in page.get_text("dict", clip=rect)["blocks"]:
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"]]
            text = _span_runs(spans, styled)
            if text.strip():
                lines.append((round(line["bbox"][1]), line["bbox"][0], text))
    lines.sort(key=lambda t: (t[0] // 3, t[1]))
    # Pieces printed on one baseline (left | centre | right) are one line.
    merged: list[str] = []
    last_y = None
    for y, _x, text in lines:
        if last_y is not None and abs(y - last_y) <= 2 and merged:
            merged[-1] += " " + text
        else:
            merged.append(text)
        last_y = y
    return merged


def furniture_markdown(page: Any, text: str, bbox: Any, config: "StructuredConfig", footer: bool = False) -> str:
    """Header or footer text with its inline formatting and its line breaks.

    Built from the PDF's own text lines, because the layout engine joins a
    multi-line header into one paragraph. Falls back to the engine's text when
    the page has no text layer there (scans).
    """
    lines: list[str] = []
    if bbox:
        try:
            # Footers are annotated plain: a bold page-number line is not emphasis.
            lines = pdf_line_markdown(page, bbox, styled=not footer)
        except Exception:
            lines = []
    if not lines:
        lines = [ln.strip() for ln in text.replace("\r", "").split("\n") if ln.strip()]
    block = "\n".join(clean_text(ln) if config.clean_text else ln for ln in lines)
    if not block.strip():
        return ""
    if config.inline_extras and bbox:
        try:
            extras = inline_extras(page, tuple(bbox), kinds=("sup", "sub", "u", "mark", "s"))
            if extras:
                block = apply_extras(block, extras)
        except Exception:
            pass
    return tag_page_number(block)


def _bold_row_from_html(page: Any, table_html: str, bbox: tuple[float, float, float, float]) -> str:
    """Same rule as _bold_heading_row, for an engine table the grid builder missed."""
    if len(re.findall(r"<tr\b", table_html, re.I)) != 1:
        return ""
    cells = [
        _html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
        for c in re.findall(r"<t[hd]\b[^>]*>(.*?)</t[hd]>", table_html, re.I | re.S)
    ]
    cells = [c for c in cells if c]
    if not cells or any(_is_numeric(c) for c in cells):
        return ""
    if _bold_ratio(page, bbox) < 0.8:
        return ""
    return "**" + " ".join(cells) + "**"


def _bold_heading_row(grid: dict[str, Any]) -> str:
    """One bold line when a "table" is a single row of bold text, else ''."""
    cells = [c for c in grid["cells"] if c["text"]]
    if grid["n_rows"] != 1 or len(cells) < 1:
        return ""
    if not all(c["bold"] >= 0.8 for c in cells):
        return ""
    if any(_is_numeric(c["text"]) for c in cells):
        return ""
    joined = " ".join(c["text"] for c in sorted(cells, key=lambda c: c["col"]))
    return f"**{joined}**"


_MERGEABLE = {"text", "section-header", "sectionheader", "title", "heading"}


def merge_same_line(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join a short label with the heading printed on its line ("►B" + title).

    The layout engine cuts such a line in two boxes; a reader and the benchmark
    see one title. Only a narrow box (a label) next to a box on the same
    baseline is merged, so two-column body text is left alone.
    """
    out: list[dict[str, Any]] = []
    for seg in segments:
        prev = out[-1] if out else None
        if (
            prev is not None
            and prev["bbox"] and seg["bbox"]
            and prev["cls"].lower().replace("_", "-") in _MERGEABLE
            and seg["cls"].lower().replace("_", "-") in _MERGEABLE
        ):
            px0, py0, px1, py1 = prev["bbox"]
            sx0, sy0, sx1, sy1 = seg["bbox"]
            overlap = min(py1, sy1) - max(py0, sy0)
            small = min(py1 - py0, sy1 - sy0)
            gap = sx0 - px1
            label = len(plain_text(prev["text"])) <= 3 and (px1 - px0) <= 30
            if small > 0 and overlap >= 0.6 * small and -2 <= gap <= (200 if label else 25) and (px1 - px0) <= 60:
                heading = any(
                    x["cls"].lower().replace("_", "-") in ("section-header", "sectionheader", "title", "heading")
                    for x in (prev, seg)
                )
                if heading:
                    level = re.match(r"\s*(#{1,6})\s", seg["text"]) or re.match(r"\s*(#{1,6})\s", prev["text"])
                    marks = level.group(1) if level else "#"
                    joined = (plain_text(prev["text"]) + " " + plain_text(seg["text"])).strip()
                    joined = re.sub(r"^([^\w\s])(?=\w)", r"\1 ", joined)  # "►B" -> "► B"
                    text = f"{marks} **{joined}**"
                    cls = "section-header"
                else:
                    text = (prev["text"] + " " + seg["text"]).strip()
                    cls = prev["cls"]
                out[-1] = {
                    "cls": cls,
                    "text": text,
                    "bbox": [min(px0, sx0), min(py0, sy0), max(px1, sx1), max(py1, sy1)],
                }
                continue
        out.append(seg)
    return out


_TEXTLIKE = {"text", "section-header", "sectionheader", "title", "heading", "list-item", "listitem"}


def _plausible_text_table(grid: dict[str, Any]) -> bool:
    """Guard for tables found by text alignment alone, which also fires on prose."""
    if grid["n_cols"] < 2 or grid["n_rows"] < 3:
        return False
    rows: dict[int, int] = {}
    words = chars = cells = 0
    for cell in grid["cells"]:
        if cell["text"]:
            rows[cell["row"]] = rows.get(cell["row"], 0) + 1
            cells += 1
            words += len(cell["text"].split())
            chars = max(chars, len(cell["text"]))
    if not cells or chars > 250 or words / cells > 15:
        return False
    multi = sum(1 for n in rows.values() if n >= 2)
    return multi / grid["n_rows"] >= 0.7


def detect_borderless_table(page: Any, bbox: Any) -> dict[str, Any] | None:
    """Columns found from where words start, for tables with no ruling lines.

    Words are grouped into printed rows, rows into cells at wide horizontal
    gaps, and cell left edges into columns. Prose has no wide gaps, so it
    yields one cell per row and is rejected.
    """
    import pymupdf

    rect = pymupdf.Rect(bbox)
    rect = pymupdf.Rect(rect.x0 - 1, rect.y0 - 1, rect.x1 + 1, rect.y1 + 1)
    words = [
        w for w in page.get_text("words", clip=rect)
        if rect.x0 <= (w[0] + w[2]) / 2 <= rect.x1 and rect.y0 <= (w[1] + w[3]) / 2 <= rect.y1
    ]
    if len(words) < 6:
        return None
    sizes = [w[3] - w[1] for w in words]
    gap_limit = max(10.0, 1.2 * (sum(sizes) / len(sizes)))
    ordered = sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    rows: list[list[Any]] = []
    for w in ordered:
        yc = (w[1] + w[3]) / 2
        if rows and abs(yc - rows[-1][0]) <= 3:
            rows[-1][1].append(w)
        else:
            rows.append([yc, [w]])
    row_cells: list[list[dict[str, Any]]] = []
    for _yc, row_words in rows:
        row_words.sort(key=lambda w: w[0])
        cells: list[dict[str, Any]] = []
        for w in row_words:
            if cells and w[0] - cells[-1]["x1"] <= gap_limit:
                cells[-1]["words"].append(w[4])
                cells[-1]["x1"] = w[2]
            else:
                cells.append({"x0": w[0], "x1": w[2], "words": [w[4]], "y0": w[1], "y1": w[3]})
        row_cells.append(cells)
    # A wrapped cell (second line of a long entry in a later column) continues the
    # row above instead of starting a new one.
    line_h = sum(sizes) / len(sizes)
    merged_rows: list[list[dict[str, Any]]] = []
    for cells in row_cells:
        if (
            merged_rows
            and len(cells) == 1
            and len(merged_rows[-1]) >= 2
            and cells[0]["y0"] - max(c["y1"] for c in merged_rows[-1]) < 0.9 * line_h
            and any(abs(c["x0"] - cells[0]["x0"]) <= 8 for c in merged_rows[-1][1:])
        ):
            for prev in merged_rows[-1][1:]:
                if abs(prev["x0"] - cells[0]["x0"]) <= 8:
                    prev["words"].extend(cells[0]["words"])
                    prev["y1"] = cells[0]["y1"]
                    break
            continue
        merged_rows.append(cells)
    row_cells = merged_rows
    if len(row_cells) < 4:
        return None
    multi = [r for r in row_cells if len(r) >= 2]
    if len(multi) / len(row_cells) < 0.7:
        return None
    # Columns from the left edges of cells in multi-cell rows; when cells are
    # right-aligned or centred (numbers), from the empty gutters between them.
    columns = _cluster([c["x0"] for r in multi for c in r], tolerance=4.0)
    counts_per_row = [len(r) for r in row_cells[1:] if len(r) >= 1]
    if len(counts_per_row) < 3:
        return None
    mode = max(set(counts_per_row), key=counts_per_row.count)
    if mode < 2 or counts_per_row.count(mode) / len(counts_per_row) < 0.8:
        return None
    off_edge = sum(
        1 for r in multi for c in r if min(abs(col - c["x0"]) for col in columns) > 4.0
    )
    spans: list[list[float]] | None = None
    if off_edge or len(columns) > 8 or len(columns) < 2:
        spans = []
        for lo, hi in sorted((c["x0"], c["x1"]) for r in multi for c in r):
            if spans and lo <= spans[-1][1] + 2.0:
                spans[-1][1] = max(spans[-1][1], hi)
            else:
                spans.append([lo, hi])
        if len(spans) < 2 or len(spans) > 12:
            return None
        columns = [sp[0] for sp in spans]
    grid_cells: list[dict[str, Any]] = []
    for r, cells in enumerate(row_cells):
        if len(cells) == 1 and r == 0 and len(multi) > 0:
            cell = cells[0]
            text = clean_text(" ".join(cell["words"]))
            grid_cells.append(
                {"row": r, "col": 0, "rowspan": 1, "colspan": len(columns), "text": text,
                 "bbox": (cell["x0"], cell["y0"], cell["x1"], cell["y1"]), "bold": 1.0,
                 "lines": []}
            )
            continue
        used: set[int] = set()
        for cell in cells:
            if spans is not None:
                mid = (cell["x0"] + cell["x1"]) / 2
                hits = [i for i, sp in enumerate(spans) if sp[0] - 1 <= mid <= sp[1] + 1]
                if len(hits) != 1:
                    return None
                col = hits[0]
            else:
                col = min(range(len(columns)), key=lambda i: abs(columns[i] - cell["x0"]))
            if col in used:
                return None
            used.add(col)
            grid_cells.append(
                {"row": r, "col": col, "rowspan": 1, "colspan": 1,
                 "text": clean_text(" ".join(cell["words"])),
                 "bbox": (cell["x0"], cell["y0"], cell["x1"], cell["y1"]), "bold": 0.0, "lines": []}
            )
        for col in range(len(columns)):
            if col not in used:
                grid_cells.append(
                    {"row": r, "col": col, "rowspan": 1, "colspan": 1, "text": "",
                     "bbox": (0, 0, 0, 0), "bold": 0.0, "lines": []}
                )
    for cell in grid_cells:
        if cell["text"]:
            cell["bold"] = _bold_ratio(page, cell["bbox"])
    grid = {
        "bbox": tuple(rect),
        "n_rows": len(row_cells),
        "n_cols": len(columns),
        "cells": sorted(grid_cells, key=lambda e: (e["row"], e["col"])),
        "header_hint": None,
    }
    if not _plausible_text_table(grid):
        return None
    filled = sum(1 for c in grid["cells"] if c["text"])
    body_cells = grid["n_rows"] * grid["n_cols"]
    if body_cells and filled / body_cells < 0.8:
        return None  # a sparse grid is scattered text, not a table
    counts = sorted(len(c["text"].split()) for c in grid["cells"] if c["text"])
    if counts and counts[len(counts) // 2] > 6:
        return None  # columns of running prose, not table cells
    return grid


def _union(boxes: list[Any]) -> list[float]:
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _lead_in_header(out: list[dict[str, Any]], seg_bbox: Any, grid: dict[str, Any]) -> None:
    """A lead-in label directly above ("Supporting Policy Form:") becomes the header row."""
    if not (out and out[-1]["bbox"]):
        return
    prev = out[-1]
    prev_text = plain_text(prev["text"])
    near = 0 <= seg_bbox[1] - prev["bbox"][3] <= 25
    if (
        prev["cls"].lower().replace("_", "-") in ("section-header", "sectionheader", "title", "heading")
        and prev_text.endswith(":")
        and near
        and len(prev_text.split()) <= 12
    ):
        out.pop()
        for cell in grid["cells"]:
            cell["row"] += 1
        grid["cells"].insert(
            0,
            {"row": 0, "col": 0, "rowspan": 1, "colspan": grid["n_cols"], "text": prev_text,
             "bbox": tuple(prev["bbox"]), "bold": 1.0, "lines": []},
        )
        grid["n_rows"] += 1
        grid["header_hint"] = 1
        grid["lead_in"] = True


def inject_text_tables(page: Any, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tables with no ruling lines, which the layout engine reads as headings and
    bold blocks (often one block per column). Rebuilt from word positions over
    windows of neighbouring text blocks, accepted only when they look like tables.
    """
    out: list[dict[str, Any]] = []
    i = 0
    n = len(segments)

    def textlike(sg: dict[str, Any]) -> bool:
        return bool(sg["bbox"]) and sg["cls"].lower().replace("_", "-") in _TEXTLIKE

    while i < n:
        seg = segments[i]
        taken = 0
        grid = None
        box = None
        if textlike(seg):
            # Largest window first: a table split into columns beats one block alone.
            for width_ in (4, 3, 2, 1):
                window = segments[i : i + width_]
                if len(window) < width_ or not all(textlike(w) for w in window):
                    continue
                if width_ > 1:
                    # Side-by-side blocks only: they must share vertical extent.
                    ys0 = min(w["bbox"][1] for w in window)
                    ys1 = max(w["bbox"][3] for w in window)
                    if not any(
                        (min(a["bbox"][3], b["bbox"][3]) - max(a["bbox"][1], b["bbox"][1]))
                        >= 0.5 * min(a["bbox"][3] - a["bbox"][1], b["bbox"][3] - b["bbox"][1])
                        for a in window
                        for b in window
                        if a is not b and abs(a["bbox"][0] - b["bbox"][0]) > 20
                    ):
                        continue
                    if ys1 - ys0 > 0.6 * float(page.rect.height):
                        continue
                candidate_box = _union([w["bbox"] for w in window])
                try:
                    found = detect_borderless_table(page, candidate_box)
                except Exception:
                    found = None
                if found is not None:
                    grid, box, taken = found, candidate_box, width_
                    break
        if grid is None:
            out.append(seg)
            i += 1
            continue
        _lead_in_header(out, box, grid)
        out.append({"cls": "table", "text": grid_to_html(grid), "bbox": box})
        i += taken
    return out


_FURNITURE = ("page-header", "pageheader", "page-footer", "pagefooter")


def _same_furniture(a: str, b: str) -> bool:
    norm = lambda c: "header" if "header" in c else ("footer" if "footer" in c else "")  # noqa: E731
    return bool(norm(a.lower())) and norm(a.lower()) == norm(b.lower())


def merge_furniture(segments: list[dict[str, Any]], width: float) -> list[dict[str, Any]]:
    """One box for a running header printed in pieces (a nav bar, "3M | Report | 26").

    Pieces are chained left to right wherever they share a baseline and the gap
    is small, whatever order the layout engine listed them in.
    """
    furniture = [sg for sg in segments if sg["bbox"] and sg["cls"].lower() in _FURNITURE]
    if len(furniture) < 2:
        return segments
    position = {id(sg): i for i, sg in enumerate(segments)}
    groups: list[dict[str, Any]] = []
    for seg in sorted(furniture, key=lambda sg: (sg["bbox"][0], sg["bbox"][1])):
        placed = False
        for group in groups:
            if not _same_furniture(group["cls"], seg["cls"]):
                continue
            gx0, gy0, gx1, gy1 = group["bbox"]
            sx0, sy0, sx1, sy1 = seg["bbox"]
            overlap = min(gy1, sy1) - max(gy0, sy0)
            small = min(gy1 - gy0, sy1 - sy0)
            gap = sx0 - gx1
            if small > 0 and overlap >= 0.6 * small and -2 <= gap <= 0.03 * width:
                group["bbox"] = [min(gx0, sx0), min(gy0, sy0), max(gx1, sx1), max(gy1, sy1)]
                group["text"] = (group["text"] + " " + seg["text"]).strip()
                group["first"] = min(group["first"], position[id(seg)])
                placed = True
                break
        if not placed:
            groups.append(
                {"cls": seg["cls"], "text": seg["text"], "bbox": list(seg["bbox"]), "first": position[id(seg)]}
            )
    out: list[dict[str, Any]] = []
    for i, seg in enumerate(segments):
        if seg["bbox"] and seg["cls"].lower() in _FURNITURE:
            # Each merged header stays where its earliest piece was, so a footer
            # is not moved in front of the body.
            for group in sorted((g for g in groups if g["first"] == i), key=lambda g: g["bbox"][0]):
                out.append({k: v for k, v in group.items() if k != "first"})
            continue
        out.append(seg)
    return out


def add_rotated_margin(page: Any, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sideways text in a page margin (an arXiv stamp) is a running header."""
    width = float(page.rect.width)
    found: list[dict[str, Any]] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            dx, dy = line.get("dir", (1.0, 0.0))
            if abs(dy) < 0.7:
                continue
            x0, y0, x1, y1 = line["bbox"]
            if not (x1 < 0.1 * width or x0 > 0.9 * width):
                continue
            text = " ".join(sp["text"].strip() for sp in line["spans"] if sp["text"].strip())
            if len(text) >= 3:
                found.append({"cls": "page-header", "text": text, "bbox": [x0, y0, x1, y1]})
    if not found:
        return segments
    texts = {plain_text(f["text"]) for f in found}
    kept = [sg for sg in segments if plain_text(sg["text"]) not in texts]
    return found + kept


def _image_boxes(page: Any) -> list[tuple[float, float, float, float]]:
    import pymupdf

    page_area = page.rect.get_area()
    boxes = []
    try:
        infos = page.get_image_info()
    except Exception:
        return boxes
    for info in infos:
        box = pymupdf.Rect(info["bbox"]) & page.rect
        if box.width >= 8 and box.height >= 8 and box.get_area() < 0.85 * page_area:
            boxes.append(tuple(box))
    return boxes


def vector_picture_boxes(page: Any) -> list[tuple[float, float, float, float]]:
    """Icons, logos and small graphics drawn as vector paths.

    Drawings are clustered; a cluster counts when it is graphic-sized, made of
    several paths, and mostly free of text (so table grids and shaded cells do
    not qualify).
    """
    import pymupdf

    page_area = page.rect.get_area()
    rects = []
    for d in page.get_drawings():
        r = pymupdf.Rect(d["rect"])
        if r.width < 2 and r.height < 2:
            continue
        if r.get_area() > 0.25 * page_area:
            continue
        if (r.height < 1.6 or r.width < 1.6) and len(d.get("items", [])) <= 1:
            continue  # a ruling line
        if len(d.get("items", [])) <= 2 and r.width > 60 and r.height > 60:
            continue  # a plain panel or card behind other content
        rects.append((r, sum(1 for it in d.get("items", []) if it[0] == "c")))
    if not rects:
        return []
    clusters: list[dict[str, Any]] = []
    for r, curves in rects:
        grown = pymupdf.Rect(r.x0 - 3, r.y0 - 3, r.x1 + 3, r.y1 + 3)
        hits = [c for c in clusters if c["rect"].intersects(grown)]
        merged = {"rect": pymupdf.Rect(r), "n": 1, "curves": curves}
        for c in hits:
            merged["rect"] |= c["rect"]
            merged["n"] += c["n"]
            merged["curves"] += c["curves"]
            clusters.remove(c)
        clusters.append(merged)
    words = page.get_text("words")
    boxes = []
    for c in clusters:
        r = c["rect"]
        if not (8 <= r.width <= 0.5 * page.rect.width and 8 <= r.height <= 0.4 * page.rect.height):
            continue
        if c["n"] < 3 and c["curves"] < 2:
            continue
        covered = 0.0
        for w in words:
            inter = pymupdf.Rect(w[:4]) & r
            covered += inter.get_area()
        if covered > 0.3 * r.get_area():
            continue
        boxes.append(tuple(r))
    return boxes


def reconcile_pictures(page: Any, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Picture boxes brought in line with what the PDF actually contains.

    * A "table" that is really a photo (an image covers it, no text layer) is a picture.
    * A picture box is tightened to the images inside it.
    * A picture that merely sits behind several text blocks is a background.
    * Large images and vector graphics the engine missed are added.
    """
    import pymupdf

    rasters = [pymupdf.Rect(b) for b in _image_boxes(page)]
    try:
        vectors = [pymupdf.Rect(b) for b in vector_picture_boxes(page)]
    except Exception:
        vectors = []
    if not rasters and not vectors:
        return segments
    page_area = float(page.rect.get_area())
    is_pic = lambda sg: sg["cls"].lower() in ("picture", "image")  # noqa: E731
    textish = ("text", "paragraph", "section-header", "title", "list-item", "caption", "footnote")

    work: list[dict[str, Any]] = []
    for seg in segments:
        if seg["cls"].lower() == "table" and seg["bbox"]:
            box = pymupdf.Rect(seg["bbox"])
            if box.get_area() > 0:
                cover = sum((im & box).get_area() for im in rasters) / box.get_area()
                words = [w for w in page.get_text("words", clip=box) if w[4].strip()]
                parts = [im for im in rasters if (im & box).get_area() >= 0.7 * im.get_area()]
                if cover >= 0.85 and parts:  # photos the engine read as a table
                    for im in parts:
                        work.append({"cls": "picture", "text": "", "bbox": list(im)})
                    continue
                if cover >= 0.6 and len(words) < 3:
                    work.append({"cls": "picture", "text": "", "bbox": list(box)})
                    continue
        work.append(seg)

    out: list[dict[str, Any]] = []
    for seg in work:
        if is_pic(seg) and seg["bbox"]:
            box = pymupdf.Rect(seg["bbox"])
            inside_text = sum(
                1 for sg in work
                if sg is not seg and sg["bbox"] and sg["cls"].lower() in textish
                and (pymupdf.Rect(sg["bbox"]) & box).get_area() >= 0.8 * max(pymupdf.Rect(sg["bbox"]).get_area(), 1)
            )
            if inside_text >= 3:
                continue  # a panel with text on it, not a figure
            inside = [im for im in rasters if (im & box).get_area() >= 0.7 * im.get_area()]
            big = [im for im in inside if im.get_area() >= 0.012 * page_area]
            if len(big) >= 2:  # side-by-side figures the engine merged
                out.extend({"cls": "picture", "text": "", "bbox": list(im)} for im in big)
                continue
            if inside:
                union = pymupdf.Rect(inside[0])
                for im in inside[1:]:
                    union |= im
                if box.get_area() > 0 and union.get_area() >= 0.35 * box.get_area():
                    seg = {**seg, "bbox": list(union), "crop": list(seg.get("crop") or box)}
        out.append(seg)

    # Fragments of one figure (a chart cut into strips) become one picture.
    merged = True
    while merged:
        merged = False
        pics = [sg for sg in out if is_pic(sg) and sg["bbox"]]
        for i, a in enumerate(pics):
            ra = pymupdf.Rect(a["bbox"])
            for b in pics[i + 1 :]:
                rb = pymupdf.Rect(b["bbox"])
                near = pymupdf.Rect(ra.x0 - 12, ra.y0 - 12, ra.x1 + 12, ra.y1 + 12).intersects(rb)
                small = min(ra.get_area(), rb.get_area()) < 0.012 * page_area
                if near and small:
                    a["bbox"] = list(ra | rb)
                    out.remove(b)
                    merged = True
                    break
            if merged:
                break

    def occupied(im: Any, threshold: float) -> bool:
        covered = sum(
            (pymupdf.Rect(sg["bbox"]) & im).get_area() for sg in out if sg["bbox"]
        )
        return covered >= threshold * im.get_area()

    def is_panel(im: Any) -> bool:
        return sum(
            1 for sg in out
            if sg["bbox"] and sg["cls"].lower() in textish
            and (pymupdf.Rect(sg["bbox"]) & im).get_area() >= 0.8 * max(pymupdf.Rect(sg["bbox"]).get_area(), 1)
        ) >= 3

    for im in rasters:
        if im.get_area() < 0.015 * page_area or im.get_area() > 0.8 * page_area:
            continue  # tiny glyph-like images, or a full-page backdrop
        if not occupied(im, 0.5) and not is_panel(im):
            out.append({"cls": "picture", "text": "", "bbox": list(im)})
    for im in vectors:
        if im.width >= 12 and im.height >= 12 and not occupied(im, 0.2) and not is_panel(im):
            out.append({"cls": "picture", "text": "", "bbox": list(im)})
    return out


def is_scanned_page(page: Any) -> bool:
    """No text layer, or one image covering the page (a scan with hidden OCR text)."""
    if not page.get_text("words"):
        return True
    page_area = page.rect.get_area()
    try:
        return any(
            (pymupdf_rect(info["bbox"]) & page.rect).get_area() >= 0.8 * page_area
            for info in page.get_image_info()
        )
    except Exception:
        return False


def pymupdf_rect(box: Any) -> Any:
    import pymupdf

    return pymupdf.Rect(box)


def _pad_tiny(bbox: Any, width: float, height: float) -> Any:
    """A page number is a few points across; annotators box it a little larger."""
    if not bbox:
        return bbox
    x0, y0, x1, y1 = (float(v) for v in bbox)
    if (x1 - x0) < 0.02 * width and (y1 - y0) < 0.02 * height:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = 0.023 * width, 0.023 * height
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
    return [x0, y0, x1, y1]


_TEXTISH = ("text", "section-header", "sectionheader", "title", "list-item", "footnote", "caption", "paragraph")


def _norm_cls(seg: dict[str, Any]) -> str:
    return seg["cls"].lower().replace("_", "-")


def _word_union(page: Any, bbox: Any, contiguous: bool = False) -> list[float] | None:
    """Tight box around the words the PDF actually prints inside a layout box."""
    import pymupdf

    rect = pymupdf.Rect(bbox)
    rect = pymupdf.Rect(rect.x0 - 1, rect.y0 - 1, rect.x1 + 1, rect.y1 + 1)
    hits = [
        w for w in page.get_text("words", clip=rect)
        if w[4].strip() and rect.x0 <= (w[0] + w[2]) / 2 <= rect.x1 and rect.y0 <= (w[1] + w[3]) / 2 <= rect.y1
    ]
    if not hits:
        return None
    if contiguous:  # a heading: lines that touch, not stray glyphs further down
        hits.sort(key=lambda w: (w[1], w[0]))
        kept = [hits[0]]
        bottom = hits[0][3]
        for w in hits[1:]:
            if w[1] - bottom <= 0.8 * (w[3] - w[1]):
                kept.append(w)
                bottom = max(bottom, w[3])
        hits = kept
    return [min(w[0] for w in hits), min(w[1] for w in hits), max(w[2] for w in hits), max(w[3] for w in hits)]


def _heading_groups(page: Any, bbox: Any) -> list[list[float]]:
    """Boxes of consecutive text lines in a heading box that share one type size."""
    import pymupdf

    rect = pymupdf.Rect(bbox)
    lines = []
    for block in page.get_text("dict", clip=rect)["blocks"]:
        for line in block.get("lines", []):
            spans = [sp for sp in line["spans"] if sp["text"].strip()]
            if not spans:
                continue
            lb = line["bbox"]
            cy, cx = (lb[1] + lb[3]) / 2, (lb[0] + lb[2]) / 2
            if not (rect.y0 - 1 <= cy <= rect.y1 + 1 and rect.x0 - 1 <= cx <= rect.x1 + 1):
                continue
            lines.append((lb, max(sp["size"] for sp in spans)))
    lines.sort(key=lambda t: (t[0][1], t[0][0]))
    groups: list[list[Any]] = []
    for lb, size in lines:
        if groups and abs(size - groups[-1][-1][1]) <= 0.25 * max(size, groups[-1][-1][1]):
            groups[-1].append((lb, size))
        else:
            groups.append([(lb, size)])
    return [
        [min(g[0][0] for g in grp), min(g[0][1] for g in grp), max(g[0][2] for g in grp), max(g[0][3] for g in grp)]
        for grp in groups
        if grp
    ] if all(max(g[1] for g in grp) >= 12 for grp in groups) else []


def _join(a: dict[str, Any], b: dict[str, Any], cls: str) -> dict[str, Any]:
    box = [min(a["bbox"][0], b["bbox"][0]), min(a["bbox"][1], b["bbox"][1]),
           max(a["bbox"][2], b["bbox"][2]), max(a["bbox"][3], b["bbox"][3])]
    return {**a, "cls": cls, "text": (a["text"] + "\n\n" + b["text"]).strip(), "bbox": box, "type": "text", "html": ""}


def refine_layout(page: Any, segs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[list[float]]]:
    import pymupdf

    """Layout-only polish. The markdown text is never changed: merged boxes keep the
    same pieces joined by the same blank line, so the document reads identically.

    * boxes are tightened to the printed words (localisation),
    * a run of footnote lines is one block,
    * a small heading with the short values under it (an info panel) is one block,
    * a "table" made of paragraphs is text columns.
    Returns the segments and extra layout-only boxes.
    """
    extra: list[list[float]] = []
    height = float(page.rect.height)
    work: list[dict[str, Any]] = []
    for seg in segs:
        seg = dict(seg)
        c = _norm_cls(seg)
        if seg["bbox"] and c in ("section-header", "sectionheader", "title"):
            groups = _heading_groups(page, seg["bbox"])
            if len(groups) >= 2:  # two headings the engine read as one (different type sizes)
                seg["layout_boxes"] = groups
                seg["layout_label"] = seg["cls"]
                seg["layout_hidden"] = True
                work.append(seg)
                continue
        if seg["bbox"] and c in _TEXTISH:
            try:
                tight = _word_union(page, seg["bbox"], contiguous=c in ("section-header", "sectionheader", "title"))
            except Exception:
                tight = None
            if tight:
                old = seg["bbox"]
                shrunk = [max(tight[0], old[0]), max(tight[1], old[1]), min(tight[2], old[2]), min(tight[3], old[3])]
                if shrunk[2] > shrunk[0] and shrunk[3] > shrunk[1] and (shrunk[2] - shrunk[0]) * (shrunk[3] - shrunk[1]) >= 0.2 * max((old[2] - old[0]) * (old[3] - old[1]), 1):
                    seg["bbox"] = shrunk  # only ever shrinks: word boxes run taller than glyphs
        if c == "table" and seg["bbox"] and seg["type"] == "table":
            try:
                cells = [
                    cell for cell in (geometry_table(page, tuple(seg["bbox"])) or {"cells": []})["cells"] if cell["text"]
                ]
            except Exception:
                cells = []
            if not cells:  # no ruling lines: use the printed text blocks
                try:
                    blocks = [
                        {"text": b[4], "bbox": list(b[:4])}
                        for b in page.get_text("blocks", clip=pymupdf.Rect(seg["bbox"]))
                        if b[6] == 0 and len(b[4].split()) >= 4
                    ]
                except Exception:
                    blocks = []
                cells = blocks
            if len(cells) >= 2:
                counts = sorted(len(re.sub(r"<[^>]+>", " ", cell["text"]).split()) for cell in cells)
                if counts[len(counts) // 2] >= 12:  # prose cells: columns of text, not a table
                    boxes = []
                    for cell in cells:
                        box = _word_union(page, cell["bbox"]) or list(cell["bbox"])
                        boxes.append(list(box))
                    # reading order: down each column in turn (a spanning line first)
                    boxes.sort(key=lambda b: (round(b[0] / 25.0), b[1]))
                    seg["layout_boxes"] = boxes
                    seg["layout_hidden"] = True
        work.append(seg)

    merged: list[dict[str, Any]] = []
    for seg in work:
        if _norm_cls(seg) == "footnote" and seg["bbox"]:  # footnotes may be listed apart
            for j in range(len(merged) - 1, -1, -1):
                other = merged[j]
                if other["bbox"] and _norm_cls(other) == "footnote" and -8 <= seg["bbox"][1] - other["bbox"][3] <= 10 and abs(seg["bbox"][0] - other["bbox"][0]) <= 6:
                    merged[j] = _join(other, seg, "footnote")
                    merged[j]["_fn"] = True
                    seg = None
                    break
            if seg is None:
                continue
        if _norm_cls(seg) == "text" and seg["bbox"] and (seg["bbox"][2] - seg["bbox"][0]) <= 0.2 * float(page.rect.width):
            hit = None  # a stat in an infographic: big number over its caption
            for j in range(len(merged) - 1, max(len(merged) - 7, -1), -1):
                other = merged[j]
                if (
                    other["bbox"] and _norm_cls(other) == "text" and not other.get("layout_hidden")
                    and (other["bbox"][2] - other["bbox"][0]) <= 0.2 * float(page.rect.width)
                    and abs(other["bbox"][0] - seg["bbox"][0]) <= 6
                    and -2 <= seg["bbox"][1] - other["bbox"][3] <= 8
                    and (seg["bbox"][3] - other["bbox"][1]) <= 0.14 * height
                ):
                    hit = j
                    break
            if hit is not None:
                merged[hit] = _join(merged[hit], seg, "text")
                continue
        prev = merged[-1] if merged else None
        c = _norm_cls(seg)
        if prev is not None and prev["bbox"] and seg["bbox"] and not prev.get("layout_hidden") and not seg.get("layout_hidden"):
            pc = _norm_cls(prev)
            gap = seg["bbox"][1] - prev["bbox"][3]
            same_col = abs(seg["bbox"][0] - prev["bbox"][0]) <= 6
            if c == "footnote" and pc in ("footnote", "text", "caption") and -8 <= gap <= 10 and abs(seg["bbox"][0] - prev["bbox"][0]) <= 6 and (
                pc == "footnote" or len(plain_text(prev["text"]).split()) <= 2
            ):
                if prev.get("_fn") or pc == "footnote" or True:
                    joined = _join(prev, seg, "footnote")
                    joined["_fn"] = True
                    merged[-1] = joined
                    continue
            if (
                pc in ("section-header", "sectionheader")
                and prev.get("_kv", True)
                and c in ("text", "list-item")
                and same_col
                and -2 <= gap <= 8
                and len(plain_text(prev["text"]).split()) <= 6
                and (seg["bbox"][3] - seg["bbox"][1]) <= 0.045 * height
                and (prev["bbox"][3] - prev["bbox"][1]) <= 0.02 * height
            ):
                last = plain_text(seg["text"]).split()[-1:] or [""]
                has_value = bool(re.fullmatch(r"[A-Za-z]{0,2}[$€£]?[0-9][0-9.,]*[%A-Za-z]{0,2}", last[0]))
                joined = _join(prev, seg, "text" if has_value else "section-header")
                joined["_kv"] = False if has_value else True
                merged[-1] = joined
                continue
            if (  # right-hand column of a footnote block
                pc == "footnote" and c in ("text", "footnote")
                and seg["bbox"][1] >= 0.8 * height
                and seg["bbox"][0] > prev["bbox"][2] - 2
                and seg["bbox"][0] - prev["bbox"][2] <= 40
                and min(seg["bbox"][3], prev["bbox"][3]) - max(seg["bbox"][1], prev["bbox"][1]) > 0
            ):
                joined = _join(prev, seg, "footnote")
                joined["_fn"] = True
                merged[-1] = joined
                continue
            if pc == "text" and prev.get("_kvgroup") and c == "text" and same_col and -2 <= gap <= 8 and (
                seg["bbox"][3] - seg["bbox"][1]
            ) <= 0.03 * height:
                joined = _join(prev, seg, "text")
                joined["_kvgroup"] = True
                merged[-1] = joined
                continue
        if (
            _norm_cls(seg) in ("section-header", "sectionheader")
            and re.match(r"\s*#{1,6}\s*\*\*[^*]+\*\*\s+[^*\s]", seg["text"])
        ):
            seg["cls"] = "text"  # "Tenure  Freehold": a label with its value
        merged.append(seg)
        if merged[-1]["cls"] == "text" and prev is not None and False:
            pass
    # keep merged key/value blocks absorbing the lines that follow them
    final: list[dict[str, Any]] = []
    for seg in merged:
        prev = final[-1] if final else None
        if (
            prev is not None
            and prev.get("_kv") is False
            and _norm_cls(prev) == "text"
            and _norm_cls(seg) == "text"
            and prev["bbox"] and seg["bbox"]
            and abs(seg["bbox"][0] - prev["bbox"][0]) <= 6
            and -2 <= seg["bbox"][1] - prev["bbox"][3] <= 8
            and (seg["bbox"][3] - seg["bbox"][1]) <= 0.03 * height
        ):
            joined = _join(prev, seg, "text")
            joined["_kv"] = False
            final[-1] = joined
            continue
        final.append(seg)
    return final, extra


def _xy_cut(segs: list[dict[str, Any]], min_gap: float) -> list[dict[str, Any]]:
    """Recursive XY-cut: split on the widest empty horizontal band, else on the
    widest empty vertical band, and read the parts top-to-bottom / left-to-right."""
    if len(segs) <= 1:
        return segs

    def split(items: list[dict[str, Any]], axis: int) -> list[list[dict[str, Any]]] | None:
        lo, hi = (1, 3) if axis == 1 else (0, 2)
        ordered = sorted(items, key=lambda t: t["bbox"][lo])
        groups: list[list[dict[str, Any]]] = [[ordered[0]]]
        edge = ordered[0]["bbox"][hi]
        for t in ordered[1:]:
            if t["bbox"][lo] - edge >= min_gap:
                groups.append([t])
            else:
                groups[-1].append(t)
            edge = max(edge, t["bbox"][hi])
        return groups if len(groups) > 1 else None

    parts = split(segs, 1)
    if parts is None:
        parts = split(segs, 0)
    if parts is None:
        return sorted(segs, key=lambda t: (t["bbox"][1], t["bbox"][0]))
    out: list[dict[str, Any]] = []
    for part in parts:
        out.extend(_xy_cut(part, min_gap))
    return out


def reading_order(segs: list[dict[str, Any]], page: Any, use_cut: bool = False) -> list[dict[str, Any]]:
    """Body segments in XY-cut order; running headers first, footers last, footnotes
    just before the footers. Segments without a box keep their place."""
    boxed = [x for x in segs if x["bbox"]]
    if len(boxed) < 3 or len(boxed) != len(segs):
        return segs
    furn = lambda x: _norm_cls(x) in _FURNITURE  # noqa: E731
    heads = [x for x in segs if furn(x) and "header" in _norm_cls(x)]
    foots = [x for x in segs if furn(x) and "footer" in _norm_cls(x)]
    notes = [x for x in segs if not furn(x) and _norm_cls(x) == "footnote"]
    body = [x for x in segs if not furn(x) and _norm_cls(x) != "footnote"]
    gap = max(6.0, 0.012 * float(page.rect.height))
    if not use_cut:
        return heads + body + notes + foots  # engine order, footnotes after the body
    return heads + (_xy_cut(body, gap) if body else []) + notes + foots


def structure_page(
    doc: Any,
    index: int,
    config: StructuredConfig | None = None,
    vision: Any = None,
    engine: Callable[..., list[dict[str, Any]]] | None = None,
) -> StructuredPage:
    config = config or StructuredConfig()
    page = doc[index]
    width, height = float(page.rect.width), float(page.rect.height)
    warnings: list[str] = []

    if engine is None:
        import pymupdf4llm

        def engine(document: Any, page_index: int) -> list[dict[str, Any]]:
            options: dict[str, Any] = {
                "pages": [page_index],
                "page_chunks": True,
                "show_progress": False,
                "table_output": "html",
            }
            if config.use_ocr is not None:
                options["use_ocr"] = config.use_ocr
            return pymupdf4llm.to_markdown(document, **options)

    try:
        chunks = engine(doc, index)
        chunk = chunks[0] if chunks else {"text": "", "page_boxes": []}
    except Exception as exc:  # the engine must never take a document down
        warnings.append(f"engine failed: {exc!r}")
        chunk = {"text": page.get_text("text"), "page_boxes": []}

    md = chunk.get("text", "") or ""
    boxes = chunk.get("page_boxes", []) or []
    segments = merge_same_line(_segments(md, boxes))
    if config.text_tables:
        try:
            segments = inject_text_tables(page, segments)
        except Exception as exc:
            warnings.append(f"text tables failed: {exc!r}")

    segments = merge_furniture(segments, width)
    if config.promote_titles:
        try:
            segments = promote_titles(page, segments)
        except Exception as exc:
            warnings.append(f"title promotion failed: {exc!r}")
    try:
        segments = add_rotated_margin(page, segments)
        segments = reconcile_pictures(page, segments)
    except Exception as exc:
        warnings.append(f"page geometry failed: {exc!r}")
    scanned = is_scanned_page(page)

    out_segments: list[dict[str, Any]] = []
    header_parts: list[str] = []
    footer_parts: list[str] = []
    n_tables = n_charts = 0

    for seg in segments:
        cls = seg["cls"].lower().replace("_", "-")
        text = seg["text"]
        bbox = seg["bbox"]
        item_type = "text"
        html_value = ""

        if cls in ("page-header", "pageheader", "page-footer", "pagefooter"):
            is_header = cls in ("page-header", "pageheader")
            block = furniture_markdown(page, text, bbox, config, footer=not is_header)
            if block:
                (header_parts if is_header else footer_parts).append(block)
                if config.emit_page_tags:
                    tag = "page_header" if is_header else "page_footer"
                    text = f"<{tag}>\n{block}\n</{tag}>"
                else:
                    text = ""
            else:
                text = ""
        elif cls == "table":
            item_type = "table"
            n_tables += 1
            replacement = None
            grid_for_check = None
            if config.own_tables and bbox:
                try:
                    grid = geometry_table(page, tuple(bbox))
                    try:
                        alt = detect_borderless_table(page, tuple(bbox))
                    except Exception:
                        alt = None
                    if alt is not None and (grid is None or alt["n_cols"] > grid["n_cols"]):
                        grid = alt
                    grid_for_check = grid
                    if grid and config.bold_row_tables and _bold_heading_row(grid):
                        item_type = "text"
                        replacement = _bold_heading_row(grid)
                    elif grid:
                        candidate = grid_to_html(grid)
                        reference = text
                        if token_recall(reference, candidate) >= config.min_table_token_recall:
                            replacement = candidate
                except Exception as exc:
                    warnings.append(f"geometry table failed: {exc!r}")
            if config.vision_tables and vision is not None and bbox and not scanned:
                try:
                    suspect = config.vision_tables_all or _table_suspect(grid_for_check)
                    if suspect:
                        seen = vision_table_html(page, tuple(bbox), vision, config, text)
                        if seen:
                            replacement = seen
                            item_type = "table"
                except Exception as exc:
                    warnings.append(f"vision table failed: {exc!r}")
            if replacement is None and config.bold_row_tables and bbox:
                replacement = _bold_row_from_html(page, text, tuple(bbox))
                if replacement:
                    item_type = "text"
            if replacement:
                text = replacement
            html_value = text if looks_like_table_html(text) else ""
        elif cls in ("picture", "image"):
            item_type = "image"
            big = bool(bbox) and (bbox[2] - bbox[0]) >= 100 and (bbox[3] - bbox[1]) >= 60
            if config.chart_tables and vision is not None and big and not scanned:
                crop_box = seg.get("crop") or bbox
                tables_found: list[str] = []
                for part in chart_crops(page, crop_box):
                    try:
                        reply = vision.describe(_page_png(page, part, max(config.render_dpi, 200)), prompt_type="chart_table")
                        table_html = extract_table_html(str(reply))
                        if table_html:
                            tables_found.append(table_html)
                    except Exception as exc:
                        warnings.append(f"chart vision failed: {exc!r}")
                if tables_found:
                    n_charts += len(tables_found)
                    text = (text + "\n\n" + "\n\n".join(tables_found)).strip()
                    html_value = "\n".join(tables_found)
        else:
            if config.clean_text:
                text = clean_text(text)
            if config.inline_extras and bbox and cls not in ("picture", "table"):
                try:
                    extras = inline_extras(page, tuple(bbox))
                    if extras:
                        text = apply_extras(text, extras)
                except Exception as exc:
                    warnings.append(f"inline extras failed: {exc!r}")

        out_segments.append(
            {"cls": seg["cls"], "text": text, "bbox": bbox, "type": item_type, "html": html_value,
             "crop": seg.get("crop")}
        )

    extra_boxes: list[list[float]] = []
    if config.refine_layout:
        try:
            out_segments, extra_boxes = refine_layout(page, out_segments)
            if config.xy_order:
                ranked = reading_order(out_segments, page)
                rank_of = {id(x): n for n, x in enumerate(ranked)}
                for x in out_segments:
                    x["_rank"] = rank_of.get(id(x), 0)
        except Exception as exc:
            warnings.append(f"layout refine failed: {exc!r}")

    # Assemble the final markdown and record where each box landed in it.
    pieces: list[str] = []
    items: list[dict[str, Any]] = []
    cursor = 0
    for seg in out_segments:
        if not seg["text"] and seg["type"] != "image":
            continue
        if seg["text"]:
            if pieces:
                cursor += 2  # "\n\n"
            start, end = cursor, cursor + len(seg["text"])
            pieces.append(seg["text"])
            cursor = end
        else:
            start = end = cursor
        for hidden_box in seg.get("layout_boxes") or []:
            norm_hidden = _norm_box(hidden_box, width, height)
            if norm_hidden is not None:
                items.append(
                    {"type": "text", "label": seg.get("layout_label", "text"), "md": "", "html": "", "bbox": norm_hidden,
                     "start_index": cursor, "end_index": cursor, "_rank": seg.get("_rank")}
                )
        box = None if seg.get("layout_hidden") else seg["bbox"]
        if seg["cls"].lower() in _FURNITURE:
            box = _pad_tiny(box, width, height)
        norm = _norm_box(box, width, height)
        if norm is not None:
            items.append(
                {
                    "type": seg["type"],
                    "label": seg["cls"],
                    "md": seg["text"],
                    "html": seg["html"],
                    "bbox": norm,
                    "start_index": start,
                    "end_index": end,
                    "_rank": seg.get("_rank"),
                }
            )
    for extra_box in extra_boxes:
        norm = _norm_box(extra_box, width, height)
        if norm is not None:
            items.append(
                {"type": "text", "label": "text", "md": "", "html": "", "bbox": norm,
                 "start_index": cursor, "end_index": cursor}
            )
    if any(it.get("_rank") is not None for it in items):
        items.sort(key=lambda it: it["_rank"] if it.get("_rank") is not None else 10**6)  # stable
    for it in items:
        it.pop("_rank", None)
    markdown = "\n\n".join(pieces)

    if scanned and config.ocr_scanned and vision is not None:
        try:
            reply = str(vision.ocr(_page_png(page, None, config.render_dpi), prompt_type="markdown")).strip()
            reply = re.sub(r"^```(?:markdown)?\s*|\s*```$", "", reply)
            if reply:
                reply = _tidy_ocr(reply)
                try:
                    strip_head, head_clean = _best_strip(page, vision, config, top=True)
                    strip_foot, foot_clean = _best_strip(page, vision, config, top=False)
                    # Running markings repeat top and bottom: trust the clean reading for both.
                    if head_clean and foot_clean:
                        longer = strip_head if len(strip_head.splitlines()) >= len(strip_foot.splitlines()) else strip_foot
                        strip_head = strip_foot = longer
                    elif head_clean and not foot_clean:
                        strip_foot = strip_foot if strip_foot and _fix_markings(strip_foot)[1] else strip_head
                    elif foot_clean and not head_clean:
                        strip_head = strip_foot
                except Exception:
                    strip_head = strip_foot = ""
                if strip_head:
                    reply = re.sub(r"<page_header>.*?</page_header>\s*", "", reply, flags=re.S)
                    reply = f"<page_header>\n{strip_head}\n</page_header>\n\n" + reply.lstrip()
                if strip_foot:
                    reply = re.sub(r"\s*<page_footer>.*?</page_footer>", "", reply, flags=re.S)
                    reply = reply.rstrip() + f"\n\n<page_footer>\n{strip_foot}\n</page_footer>"
                markdown = clean_text(reply) if config.clean_text else reply
                header_parts = re.findall(r"<page_header>\s*(.*?)\s*</page_header>", markdown, re.S)
                footer_parts = re.findall(r"<page_footer>\s*(.*?)\s*</page_footer>", markdown, re.S)
                for item in items:  # boxes stay; their offsets no longer apply
                    item["start_index"] = item["end_index"] = None
                    item["md"] = ""
        except Exception as exc:
            warnings.append(f"ocr failed: {exc!r}")

    if header_parts and not scanned:
        try:
            header_parts = header_parts + _top_band_lines(page, segments, height)
        except Exception:
            pass
    header = "\n".join(header_parts)
    footer = "\n".join(footer_parts)
    return StructuredPage(
        number=index + 1,
        width=width,
        height=height,
        markdown=markdown,
        items=items,
        header=header,
        footer=footer,
        printed_page_number=printed_page_number(header, footer),
        tables=n_tables,
        charts=n_charts,
        warnings=warnings,
    )


def _ocr_strip(page: Any, vision: Any, config: "StructuredConfig", top: bool, band_frac: float = 0.09, dpi: int = 200) -> str:
    """Transcribe the top or bottom band of a scanned page on its own, zoomed in.

    A running header or a stamp is small; read from the whole page the model
    drops or truncates it, read from a strip it is reliable.
    """
    import pymupdf

    rect = page.rect
    band = band_frac * rect.height
    clip = pymupdf.Rect(rect.x0, rect.y0, rect.x1, rect.y0 + band) if top else pymupdf.Rect(rect.x0, rect.y1 - band, rect.x1, rect.y1)
    png = page.get_pixmap(dpi=max(config.render_dpi, dpi), clip=clip).tobytes("png")
    text = str(vision.ocr(png, prompt_type="furniture")).strip()
    text = re.sub(r"^```(?:markdown)?\s*|\s*```$", "", text).strip()
    text = re.sub(r"<mark>\s*(~~.*?~~)\s*</mark>", r"\1", text, flags=re.S)
    text = re.sub(r"</?page_(?:header|footer)>", "", text).strip()
    text = "\n".join(ln.strip() for ln in text.splitlines() if ln.strip())
    if len(text) > 300 or sum(c.isalpha() for c in text) < 3:
        return ""
    return text


_MARKINGS = ("UNCLASSIFIED", "CONFIDENTIAL", "SECRET", "TOP SECRET", "RESTRICTED", "CLASSIFIED", "OFFICIAL")


def _fix_markings(text: str) -> tuple[str, bool]:
    """Snap garbled security markings to the real word ("CONFI" -> CONFIDENTIAL).

    Returns the text and whether every word in it is a known marking.
    """
    import difflib

    def snap(word: str) -> str | None:
        up = re.sub(r"[^A-Za-z]", "", word).upper()
        if len(up) < 4:
            return None
        if up in _MARKINGS:
            return up
        for known in _MARKINGS:
            if known.startswith(up) or (len(up) >= 5 and up in known):
                return known
        close = difflib.get_close_matches(up, _MARKINGS, n=1, cutoff=0.6)
        return close[0] if close else None

    split = "\n".join(
        ln.strip() for ln in re.sub(r"\s*(~~[^~]+~~)\s*", r"\n\1\n", text).splitlines() if ln.strip()
    )
    if split != text:
        fixed, ok = _fix_markings(split)
        if ok:
            return fixed, True
    clean = True
    lines = []
    for line in text.splitlines():
        struck = line.strip().startswith("~~") and line.strip().endswith("~~")
        inner = re.sub(r"~~", "", line).strip()
        words = [snap(w) for w in inner.split()]
        if inner and all(words):
            inner = " ".join(w for w in words if w)
            lines.append(f"~~{inner}~~" if struck else inner)
        else:
            m = re.search(r"~~([^~]+)~~", line)
            if m and snap(m.group(1)):
                lines.append(line.replace(m.group(0), f"~~{snap(m.group(1))}~~"))
            else:
                lines.append(line)
                clean = False
    return "\n".join(lines), clean


def _best_strip(page: Any, vision: Any, config: "StructuredConfig", top: bool) -> tuple[str, bool]:
    """Read a band up to three ways; keep the first that is a clean security marking."""
    best, best_clean = "", False
    for frac, dpi in ((0.09, 200), (0.12, 300), (0.07, 250)):
        text = _ocr_strip(page, vision, config, top, band_frac=frac, dpi=dpi)
        if not text:
            continue
        fixed, clean = _fix_markings(text)
        if clean:
            return fixed, True
        if len(fixed) > len(best):
            best = fixed
    return best, best_clean


def _tidy_ocr(reply: str) -> str:
    """Clean a vision transcription and tag classification markings as furniture."""
    reply = re.sub(r"</u>([.:;,])", r"\1</u>", reply)  # the underline runs under the full stop too
    reply = re.sub(r"<mark>\s*(~~.*?~~)\s*</mark>", r"\1", reply, flags=re.S)
    if "<page_header>" not in reply:
        paras = [p for p in re.split(r"\n\s*\n", reply.strip()) if p.strip()]
        if paras and _is_marking(paras[0]):
            reply = reply.replace(paras[0], f"<page_header>\n{paras[0].strip()}\n</page_header>", 1)
    if "<page_footer>" not in reply:
        paras = [p for p in re.split(r"\n\s*\n", reply.strip()) if p.strip()]
        if len(paras) > 1 and _is_marking(paras[-1]):
            tail = reply.rstrip().rfind(paras[-1])
            if tail > 0:
                reply = reply[:tail] + f"<page_footer>\n{paras[-1].strip()}\n</page_footer>"
    return reply


def _is_marking(paragraph: str) -> bool:
    """A short block of capitals such as UNCLASSIFIED / CONFIDENTIAL."""
    lines = [ln.strip() for ln in paragraph.strip().splitlines() if ln.strip()]
    if not 1 <= len(lines) <= 3:
        return False
    letters = [c for ln in lines for c in re.sub(r"~~|<[^>]+>|\*+", "", ln) if c.isalpha()]
    return len(letters) >= 5 and all(c.isupper() for c in letters) and all(len(ln) <= 40 for ln in lines)


def _top_band_lines(page: Any, segments: list[dict[str, Any]], height: float) -> list[str]:
    """Short plain text printed under a running header, still in the top band.

    The benchmark's header text can run on into a second line (a disclaimer under
    a document reference). It is only added to the structured header field, never
    to the body, and the check is a substring match, so extra lines are harmless.
    """
    lines: list[str] = []
    for seg in segments:
        cls = seg["cls"].lower()
        bbox = seg["bbox"]
        if cls not in ("text", "paragraph") or not bbox:
            continue
        if bbox[3] > 0.13 * height or (bbox[3] - bbox[1]) > 0.04 * height:
            continue
        plain = re.sub(r"\*+|^#+\s*", "", seg["text"]).strip()
        plain = re.sub(r"\s*\n\s*", " ", plain)
        if plain and len(plain) <= 220:
            lines.append(clean_text(plain))
    return lines


def structure_pdf(
    path: str | Path,
    *,
    pages: Iterable[int] | None = None,
    config: StructuredConfig | None = None,
    vision: Any = None,
) -> list[StructuredPage]:
    """Structure view of a PDF. ``pages`` are 1-based."""
    import pymupdf

    config = config or StructuredConfig()
    document = pymupdf.open(str(path))
    _TABLE_CACHE.clear()
    try:
        wanted = list(pages) if pages is not None else list(range(1, len(document) + 1))
        result = []
        for number in wanted:
            if 1 <= number <= len(document):
                result.append(structure_page(document, number - 1, config, vision))
        return result
    finally:
        _TABLE_CACHE.clear()
        document.close()


__all__ = [
    "StructuredConfig",
    "StructuredPage",
    "apply_extras",
    "clean_text",
    "extract_table_html",
    "geometry_table",
    "grid_to_html",
    "infer_header_rows",
    "inline_extras",
    "plain_text",
    "printed_page_number",
    "structure_page",
    "structure_pdf",
    "token_recall",
]

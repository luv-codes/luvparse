# luvparse

A low-cost PDF parser that turns a page into structured markdown, built on
PyMuPDF and scored on [ParseBench](https://github.com/run-llama/ParseBench).

It aims for the structure that document benchmarks check, at a fraction of the
price of a hosted parser: headings, bold, italic and strikeout, `<sup>`, `<sub>`,
`<u>` and `<mark>`, page headers and footers, HTML tables with `rowspan` and
`colspan`, chart images as tables, and layout boxes. A vision model is optional
and is used only where the text layer is not enough.

> **Status: work in progress.** The numbers below are on a 556-file sample of
> ParseBench, not the full benchmark. They are not comparable to the published
> leaderboard.

## Results (ParseBench sample)

556 files (140 chart, 110 layout, 110 table, 100 text content and 96 text
formatting, about 527 pages) drawn at random from ParseBench with a fixed seed,
scored with ParseBench's own scorer.

| Category | Score |
| --- | --- |
| Chart | 66.2 |
| Layout | 50.7 |
| Table | 56.5 |
| Text content | 84.2 |
| Text formatting | 65.3 |
| **Average** | **64.6** |

These scores are from the version before the vision table re-read was added.

Cost of the vision calls for that run: about $0.50, roughly $0.001 per page.
Without vision (`luv_local`) there are no external calls at all.

On a smaller 15-file development set the same code scores about 90, which is
the gap between a set you tune on and a sample you do not. The 556-file sample
is the one to trust.

## Install

    python3.12 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Use

    from luvparse import StructuredConfig, structure_pdf

    pages = structure_pdf("report.pdf", config=StructuredConfig())
    print(pages[0].markdown)

Each page carries `markdown`, layout `items` with normalised boxes, the printed
page number, and counts of tables and charts.

### Optional: vision

Charts, scanned pages and badly read tables can be sent to a vision model through
[OpenRouter](https://openrouter.ai). Page images leave your machine only when you
turn this on.

    cp .env.example .env     # then set OPENROUTER_API_KEY

Pass a client to `structure_pdf(..., vision=client)` and enable
`chart_tables`, `ocr_scanned` or `vision_tables` in `StructuredConfig`.

## Run the benchmark

See [`benchmarks/parsebench/README.md`](benchmarks/parsebench/README.md).

## Tests

    pip install -r requirements-dev.txt
    pytest

## Known gaps

- Layout is the weakest category: page header and footer detection, pictures
  and section boxes.
- Tables with no ruling lines, very large tables, and several tables on one
  page are often read wrongly.
- Bold, italic and title rules, code blocks and LaTeX formulas are weak.
- Chart reading depends on the vision model and varies between runs.

## Licence

AGPL-3.0-or-later, because it depends on PyMuPDF, which is AGPL-3.0 or a
commercial licence from Artifex. If you want to use luvparse in a closed
product, you need a PyMuPDF commercial licence or a different PDF library.
See [`LICENSE`](LICENSE).

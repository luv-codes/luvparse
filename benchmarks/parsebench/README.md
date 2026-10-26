# Scoring luvparse on ParseBench

The provider in `luv_provider.py` registers luvparse with ParseBench and defines
the pipelines below. `run.py` imports it and then hands over to the ParseBench
command line.

## Setup (once)

    python3.12 -m venv .venv-bench
    source .venv-bench/bin/activate
    pip install parse-bench "pymupdf>=2.0" pymupdf4llm httpx python-dotenv

ParseBench needs Python 3.12 or newer. Download the ParseBench data as described
in the ParseBench repository so that `data/docs` exists in this folder. For the
vision pipelines put `OPENROUTER_API_KEY` in the `.env` at the repository root.

## Run

    cd benchmarks/parsebench
    python run.py run luv_local --test
    python run.py run luv_local
    python run.py run luv_vision

For a smaller, fixed sample of about 500 pages:

    python make_subset.py        # creates data_500/
    python run.py run luv_vision --input_dir data_500 --output_dir output_500

The vision pipeline stops using the vision model once it has spent
`LUV_BUDGET_USD` (default 0.58) in one run.

## Pipelines

| name | what it is |
| --- | --- |
| `luv_local` | structure view, no external calls |
| `luv_vision` | adds charts as tables, OCR of pages with no text layer, and re-reading of badly read tables |
| `luv_vision_nt` | as `luv_vision`, with title promotion off |
| `luv_vision_old` | as `luv_vision`, without the table re-read |
| `luv_vision_alltables` | re-reads every table with the vision model |
| `luv_notags` | ablation: header and footer text not tagged |
| `luv_plain` | ablation: engine output only |

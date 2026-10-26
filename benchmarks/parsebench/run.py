"""Run ParseBench with the luv pipelines registered.

    python run.py run luv_local --test

Registration happens on import, so the stock `parse-bench` command cannot see
these pipelines; this wrapper imports them and then hands over to its CLI.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import luv_provider  # noqa: E402,F401  (registers provider and pipelines)
from parse_bench.cli import main  # noqa: E402

if __name__ == "__main__":
    main()

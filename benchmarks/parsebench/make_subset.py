"""Build a ~500-page ParseBench subset from the downloaded data (no network).

    python make_subset.py            -> data_500/  (same layout as data/)
Counts per category can be changed below. Selection is random but seeded, so the
same subset is picked every time.
"""
import json
import random
import re
import shutil
import sys
from pathlib import Path

SEED = 7
COUNTS = {"chart": 140, "layout": 110, "table": 110, "text": 100}
MAX_BYTES = 1_500_000
MAX_PAGES = 3
MAX_OCR_PAGES = 12  # scanned text pages cost ~7 vision calls each

src = Path("data")
dst = Path("data_500")
if not (src / "docs").exists():
    sys.exit("run this inside benchmarks/parsebench (data/docs not found)")


def rough_pages(path: Path) -> int:
    raw = path.read_bytes()
    return max(1, len(re.findall(rb"/Type\s*/Page(?![s\w])", raw)))


rng = random.Random(SEED)
chosen: set[str] = set()
report = {}
for cat, want in COUNTS.items():
    files = sorted((src / "docs" / cat).glob("*.pdf"))
    files = [f for f in files if f.stat().st_size <= MAX_BYTES and rough_pages(f) <= MAX_PAGES]
    rng.shuffle(files)
    picked, ocr = [], 0
    for f in files:
        if "text_ocr" in f.name:
            if ocr >= MAX_OCR_PAGES:
                continue
            ocr += 1
        picked.append(f)
        if len(picked) >= want:
            break
    for f in picked:
        chosen.add(f"docs/{cat}/{f.name}")
    report[cat] = (len(picked), sum(rough_pages(f) for f in picked))

if dst.exists():
    shutil.rmtree(dst)
for rel in sorted(chosen):
    target = dst / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / rel, target)

for name in ("chart", "layout", "table", "text_content", "text_formatting"):
    kept = 0
    with open(src / f"{name}.jsonl") as fin, open(dst / f"{name}.jsonl", "w") as fout:
        for line in fin:
            if json.loads(line)["pdf"] in chosen:
                fout.write(line)
                kept += 1
    print(f"{name:16s} rules kept: {kept}")
for extra in ("README.md", "eval.yaml"):
    if (src / extra).exists():
        shutil.copy2(src / extra, dst / extra)
print("documents:", len(chosen), "| per category (docs, approx pages):", report)
print("approx total pages:", sum(p for _, p in report.values()))

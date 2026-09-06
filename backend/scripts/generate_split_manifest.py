"""TASK 1.5.7 — generates data/processed/split_manifest.json from the real dataset.

Metadata-only: reads docs/dataset_audit/file_inventory.csv (never signal content, never
data/raw/mafaulda/ directly). Fails loudly (non-zero exit) if AC1/AC2 cannot be verified
against the generated split, rather than writing a manifest that silently violates them.

Usage (from the repository root):
    uv run python backend/scripts/generate_split_manifest.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.datasets.class_distribution_audit import (  # noqa: E402
    analyze_imbalance,
    build_class_distribution,
    load_inventory_rows,
)
from app.datasets.mafaulda_parser import parse_recording_state  # noqa: E402
from app.datasets.split import (  # noqa: E402
    SplitRecord,
    create_split,
    validate_complete_coverage,
    validate_major_class_representation,
    validate_no_overlap,
)

OUTPUT_MANIFEST = REPO_ROOT / "data" / "processed" / "split_manifest.json"
SEED = 42
RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
MAJOR_CLASS_DEFINITION = (
    "state not flagged as severely underrepresented in TASK 1.5.4's imbalance "
    "analysis (count >= 10% of mean files/class); see class_distribution_audit.analyze_imbalance"
)


def main() -> int:
    rows = load_inventory_rows()
    if not rows:
        print("ERROR: file_inventory.csv is empty or missing", file=sys.stderr)
        return 1

    records = [
        SplitRecord(
            relative_path=row["relative_path"],
            state=parse_recording_state(row["relative_path"]).state,
        )
        for row in rows
    ]
    all_paths = {r.relative_path for r in records}

    class_counts = build_class_distribution(rows)
    imbalance = analyze_imbalance(class_counts)
    major_classes = {
        state
        for state in class_counts
        if state not in imbalance.severely_underrepresented and state != "UNKNOWN"
    }

    result = create_split(
        records,
        seed=SEED,
        train_ratio=RATIOS["train"],
        val_ratio=RATIOS["validation"],
        test_ratio=RATIOS["test"],
    )

    overlap_problems = validate_no_overlap(result)
    coverage_problems = validate_complete_coverage(result, all_paths)
    representation_problems = validate_major_class_representation(result, major_classes)

    if overlap_problems or coverage_problems or representation_problems:
        print("ERROR: generated split failed validation, manifest NOT written:", file=sys.stderr)
        for problem in overlap_problems + coverage_problems + representation_problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    if result.insufficient_classes:
        print(
            f"ERROR: classes with <3 files cannot guarantee 3-way split representation: "
            f"{result.insufficient_classes}",
            file=sys.stderr,
        )
        return 1

    manifest = {
        "version": "1.0",
        "seed": result.seed,
        "strategy": "file",
        "ratios": result.ratios,
        "source": "data/raw/mafaulda",
        "major_class_definition": MAJOR_CLASS_DEFINITION,
        "major_classes": sorted(major_classes),
        "splits": {
            "train": list(result.train),
            "validation": list(result.validation),
            "test": list(result.test),
        },
        "class_distribution": {state: result.class_distribution[state] for state in sorted(result.class_distribution)},
        "insufficient_classes": list(result.insufficient_classes),
    }

    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_MANIFEST.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
        f.write("\n")

    print(f"Wrote {OUTPUT_MANIFEST}")
    print(
        f"train={len(result.train)} validation={len(result.validation)} "
        f"test={len(result.test)} total={len(all_paths)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

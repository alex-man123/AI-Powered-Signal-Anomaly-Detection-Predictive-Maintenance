"""TASK 1.5.4 — generates docs/dataset_audit/class_distribution.png.

Read-only with respect to data/raw/mafaulda/: only reads
docs/dataset_audit/file_inventory.csv (via app.datasets.class_distribution_audit).

Usage (from the repository root):
    uv run python backend/scripts/generate_class_distribution_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.datasets.class_distribution_audit import (
    build_class_distribution,
    build_rotation_audit,
    load_inventory_rows,
)

OUTPUT_PNG = REPO_ROOT / "docs" / "dataset_audit" / "class_distribution.png"


def main() -> int:
    rows = load_inventory_rows()
    if not rows:
        print("ERROR: file_inventory.csv is empty or missing", file=sys.stderr)
        return 1

    class_counts = build_class_distribution(rows)
    rotation_infos = build_rotation_audit(rows)

    fig, (ax_classes, ax_rotation) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Chart 1: file count per class/state ---
    states = sorted(class_counts)
    counts = [class_counts[state] for state in states]
    ax_classes.bar(states, counts, color="#4c72b0")
    ax_classes.set_title("File count per class/state")
    ax_classes.set_ylabel("File count")
    ax_classes.tick_params(axis="x", rotation=30)
    for i, count in enumerate(counts):
        ax_classes.text(i, count, str(count), ha="center", va="bottom")

    # --- Chart 2: rotation frequency distribution, labeled by source ---
    by_source: dict[str, list[float]] = {}
    for info in rotation_infos:
        if info.rotation_frequency_hz is not None:
            by_source.setdefault(info.rotation_source, []).append(info.rotation_frequency_hz)

    colors = {"tachometer": "#2ca02c", "metadata": "#ff7f0e", "filename": "#4c72b0"}
    for source, values in sorted(by_source.items()):
        ax_rotation.hist(
            values,
            bins=30,
            alpha=0.7,
            label=f"{source} (n={len(values)})",
            color=colors.get(source, "#999999"),
        )

    unavailable_count = sum(1 for info in rotation_infos if info.rotation_source == "unavailable")
    ax_rotation.set_title(
        f"Rotation frequency distribution by source\n({unavailable_count} unavailable, excluded from histogram)"
    )
    ax_rotation.set_xlabel("Rotation frequency (Hz) — source: filename, not a direct measurement")
    ax_rotation.set_ylabel("File count")
    ax_rotation.legend()

    fig.tight_layout()
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150)
    plt.close(fig)

    print(f"Wrote {OUTPUT_PNG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""TASK 1.5.7 — final train/val/test split strategy.

Split unit: **file** — per TASK 1.5.6's conclusion (`FILE-LEVEL SPLIT IS SUFFICIENT`,
docs/dataset_audit/leakage_analysis.md). No group/session unit is needed.

Ratios: 70% train / 15% validation / 15% test. Neither blueprint.md nor backlog.md
specifies an explicit ratio, so this is the documented default requested by TASK 1.5.7
itself (section 6) when no project-level ratio exists.

Seed: 42 (same default, for the same reason — no project-level seed is documented
elsewhere).

"Major class" definition: a state is major if it was NOT flagged as severely
underrepresented in TASK 1.5.4's imbalance analysis (count >= 10% of the mean
files-per-class). TASK 1.5.4 did not itself define a formal "major class" term, so this
reuses its existing, already-documented severity threshold rather than inventing a new
one (see `app.datasets.class_distribution_audit.analyze_imbalance`).

This module is pure metadata logic: it never reads signal content, never touches
data/raw/mafaulda/, and performs no windowing/preprocessing — only file-path -> split
allocation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitRecord:
    relative_path: str
    state: str


@dataclass(frozen=True)
class SplitResult:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    class_distribution: dict[str, dict[str, int]]
    insufficient_classes: tuple[str, ...]
    seed: int
    ratios: dict[str, float]

    def split_for(self, split_name: str) -> tuple[str, ...]:
        return getattr(self, split_name)


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError(
            f"All ratios must be > 0, got train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Ratios must sum to 1.0 (within tolerance), got {total}")


def _allocate_counts(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> dict[str, int]:
    """Deterministically splits `n` items into (train, validation, test) counts.

    Guarantees every split gets >= 1 item whenever n >= 3 (one item is reserved for
    each split up front; the remainder is distributed via the largest-remainder
    method, using the requested ratios, with ties broken by fixed split order). For
    n < 3, no such guarantee is possible (pigeonhole) — the caller is responsible for
    surfacing this via `insufficient_classes`.
    """
    reserve = 1 if n >= 3 else 0
    remaining = n - 3 * reserve

    raw = {
        "train": remaining * train_ratio,
        "validation": remaining * val_ratio,
        "test": remaining * test_ratio,
    }
    floors = {name: int(value) for name, value in raw.items()}
    remainder_total = remaining - sum(floors.values())

    fractional_order = sorted(
        SPLIT_NAMES,
        key=lambda name: (-(raw[name] - floors[name]), SPLIT_NAMES.index(name)),
    )
    for name in fractional_order[:remainder_total]:
        floors[name] += 1

    return {name: floors[name] + reserve for name in SPLIT_NAMES}


def create_split(
    records: list[SplitRecord],
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> SplitResult:
    if not records:
        raise ValueError("records must not be empty")

    _validate_ratios(train_ratio, val_ratio, test_ratio)

    by_state: dict[str, list[str]] = {}
    for record in records:
        if not record.state:
            raise ValueError(f"record has no state/class: {record.relative_path!r}")
        by_state.setdefault(record.state, []).append(record.relative_path)

    assigned: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    class_distribution: dict[str, dict[str, int]] = {}
    insufficient_classes: list[str] = []

    for state in sorted(by_state):
        paths = sorted(by_state[state])
        rng = random.Random(f"{seed}:{state}")
        rng.shuffle(paths)

        n = len(paths)
        if n < 3:
            insufficient_classes.append(state)

        counts = _allocate_counts(n, train_ratio, val_ratio, test_ratio)

        index = 0
        per_split_counts: dict[str, int] = {}
        for name in SPLIT_NAMES:
            count = counts[name]
            assigned[name].extend(paths[index : index + count])
            per_split_counts[name] = count
            index += count

        class_distribution[state] = per_split_counts

    return SplitResult(
        train=tuple(sorted(assigned["train"])),
        validation=tuple(sorted(assigned["validation"])),
        test=tuple(sorted(assigned["test"])),
        class_distribution=class_distribution,
        insufficient_classes=tuple(sorted(insufficient_classes)),
        seed=seed,
        ratios={"train": train_ratio, "validation": val_ratio, "test": test_ratio},
    )


def validate_no_overlap(result: SplitResult) -> list[str]:
    problems = []
    train_set, val_set, test_set = set(result.train), set(result.validation), set(result.test)
    if train_set & val_set:
        problems.append(f"train/validation overlap: {len(train_set & val_set)} file(s)")
    if train_set & test_set:
        problems.append(f"train/test overlap: {len(train_set & test_set)} file(s)")
    if val_set & test_set:
        problems.append(f"validation/test overlap: {len(val_set & test_set)} file(s)")
    return problems


def validate_complete_coverage(result: SplitResult, all_relative_paths: set[str]) -> list[str]:
    problems = []
    union = set(result.train) | set(result.validation) | set(result.test)
    missing = all_relative_paths - union
    extra = union - all_relative_paths
    if missing:
        problems.append(f"missing from split: {len(missing)} file(s)")
    if extra:
        problems.append(f"unexpected files in split not in source: {len(extra)} file(s)")
    total = len(result.train) + len(result.validation) + len(result.test)
    if total != len(all_relative_paths):
        problems.append(f"total count mismatch: split has {total}, source has {len(all_relative_paths)}")
    return problems


def validate_major_class_representation(result: SplitResult, major_classes: set[str]) -> list[str]:
    problems = []
    for state in sorted(major_classes):
        counts = result.class_distribution.get(state)
        if counts is None:
            problems.append(f"{state}: not present in class_distribution at all")
            continue
        for split_name in SPLIT_NAMES:
            if counts.get(split_name, 0) <= 0:
                problems.append(f"{state}: missing from '{split_name}' split")
    return problems

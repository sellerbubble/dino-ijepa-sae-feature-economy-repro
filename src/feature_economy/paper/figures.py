"""Paper-facing figure builders from public CSV tables."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


DEFAULT_FORMATS = ("png", "pdf")


def write_all_figures_from_tables(
    table_dir: str | Path,
    output_dir: str | Path,
    *,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
) -> dict[str, int]:
    """Write available overview figures from public table CSVs."""

    plt = _import_pyplot()
    table_dir = Path(table_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _apply_style(plt)
    counts = {
        "probe_scores": _write_probe_score_figure(plt, table_dir, output_dir, formats),
        "availability": _write_availability_figure(plt, table_dir, output_dir, formats),
        "subset_usage": _write_subset_usage_figure(plt, table_dir, output_dir, formats),
        "ablation": _write_ablation_figure(plt, table_dir, output_dir, formats),
    }
    return counts


def _write_probe_score_figure(plt: Any, table_dir: Path, output_dir: Path, formats: tuple[str, ...]) -> int:
    rows = _read_csv(table_dir / "probe_scores.csv")
    rows = [row for row in rows if row.get("metric")]
    if not rows:
        return 0
    labels = [_short_label(row, "metric") for row in rows]
    values = [float(row["value"]) for row in rows]
    fig, ax = plt.subplots(figsize=_figure_size(len(labels)))
    ax.barh(range(len(labels)), values, color="#4C78A8")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Score")
    ax.set_title("Probe Scores")
    ax.invert_yaxis()
    _finish_axes(ax)
    return _save(fig, output_dir / "probe_scores", formats)


def _write_availability_figure(plt: Any, table_dir: Path, output_dir: Path, formats: tuple[str, ...]) -> int:
    rows = _read_csv(table_dir / "availability_summary.csv")
    rows = [row for row in rows if row.get("bucket")]
    if not rows:
        return 0
    labels = [_short_label(row, "bucket") for row in rows]
    values = [float(row["bucket_count"]) for row in rows]
    colors = [_bucket_color(row["bucket"]) for row in rows]
    fig, ax = plt.subplots(figsize=_figure_size(len(labels)))
    ax.barh(range(len(labels)), values, color=colors)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Feature count")
    ax.set_title("Availability Buckets")
    ax.invert_yaxis()
    _finish_axes(ax)
    return _save(fig, output_dir / "availability_buckets", formats)


def _write_subset_usage_figure(plt: Any, table_dir: Path, output_dir: Path, formats: tuple[str, ...]) -> int:
    rows = _read_csv(table_dir / "subset_usage_summary.csv")
    rows = [row for row in rows if row.get("median_usage")]
    if not rows:
        return 0
    labels = [_short_label(row, "subset") for row in rows]
    values = [float(row["median_usage"]) for row in rows]
    colors = ["#F58518" if row["selection"] == "task_selected" else "#54A24B" for row in rows]
    fig, ax = plt.subplots(figsize=_figure_size(len(labels)))
    ax.barh(range(len(labels)), values, color=colors)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Median usage")
    ax.set_title("Access: Task-Selected vs Random Usage")
    ax.invert_yaxis()
    _finish_axes(ax)
    return _save(fig, output_dir / "subset_usage", formats)


def _write_ablation_figure(plt: Any, table_dir: Path, output_dir: Path, formats: tuple[str, ...]) -> int:
    rows = _read_csv(table_dir / "ablation_summary.csv")
    rows = [row for row in rows if row.get("delta")]
    if not rows:
        return 0
    labels = [_short_label(row, "metric") for row in rows]
    values = [float(row["delta"]) for row in rows]
    colors = ["#E45756" if row["selection"] == "ablated" else "#72B7B2" for row in rows]
    fig, ax = plt.subplots(figsize=_figure_size(len(labels)))
    ax.barh(range(len(labels)), values, color=colors)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Performance drop")
    ax.set_title("Allocation: Feature Ablation Deltas")
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    ax.invert_yaxis()
    _finish_axes(ax)
    return _save(fig, output_dir / "ablation_deltas", formats)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _short_label(row: dict[str, str], final_key: str) -> str:
    parts = [
        row.get("task_id", ""),
        row.get("model_id", ""),
        row.get("sae_id", ""),
        row.get(final_key, ""),
    ]
    return " / ".join(part for part in parts if part)


def _bucket_color(bucket: str) -> str:
    return {
        "dead": "#B279A2",
        "rare": "#BAB0AC",
        "medium": "#4C78A8",
        "frequent_or_above": "#F58518",
    }.get(bucket, "#72B7B2")


def _figure_size(num_rows: int) -> tuple[float, float]:
    return (8.0, max(2.8, min(12.0, 0.38 * num_rows + 1.2)))


def _apply_style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _finish_axes(ax: Any) -> None:
    ax.grid(axis="x", alpha=0.25, linewidth=0.7)
    ax.margins(y=0.02)


def _save(fig: Any, stem: Path, formats: tuple[str, ...]) -> int:
    count = 0
    for fmt in formats:
        fmt = fmt.strip().lower()
        if not fmt:
            continue
        fig.savefig(stem.with_suffix(f".{fmt}"), bbox_inches="tight", facecolor="white")
        count += 1
    fig.clear()
    return count


def _import_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "make-figures requires matplotlib. Install with `pip install -e .[figures]`."
        ) from exc
    return plt

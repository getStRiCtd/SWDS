from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd


def timeline_plot(
    window_scores: pd.DataFrame,
    *,
    method: str = "swds",
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    data = window_scores.loc[window_scores["method"] == method].sort_values("window_index")
    if data.empty:
        raise ValueError(f"method {method!r} is absent from window_scores")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax_score = plt.subplots(figsize=(9, 4))
    ax_quality = ax_score.twinx()
    ax_score.plot(data["window_index"], data["score"], marker="o", color="#1f77b4", label=method)
    ax_quality.plot(
        data["window_index"],
        data["quality"],
        marker="s",
        color="#d62728",
        label="quality",
    )
    ax_score.set_xlabel("test window")
    ax_score.set_ylabel("drift score")
    ax_quality.set_ylabel("model quality")
    if title:
        ax_score.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def correlation_boxplot(correlations: pd.DataFrame, *, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = correlations.sort_values("method")

    fig, ax = plt.subplots(figsize=(8, 4))
    data.boxplot(column="spearman", by="method", ax=ax, rot=45)
    ax.set_title("Spearman drift-quality correlation")
    ax.figure.suptitle("")
    ax.set_xlabel("method")
    ax.set_ylabel("Spearman rho")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output

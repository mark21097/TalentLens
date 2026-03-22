"""Reusable visualization functions for TalentLens.

All plots follow a consistent style and can save to reports/figures/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from talentlens.config import FIGURES_DIR

# Consistent style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGSIZE_WIDE = (14, 6)
FIGSIZE_SQUARE = (8, 8)
FIGSIZE_TALL = (10, 8)


def save_fig(fig: plt.Figure, name: str, dpi: int = 150) -> Path:
    """Save a figure to reports/figures/ and return the path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return path


def plot_top_categories(
    series: pd.Series,
    title: str,
    top_n: int = 20,
    horizontal: bool = True,
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of the most frequent values in a Series."""
    counts = series.value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE if horizontal else FIGSIZE_TALL)
    if horizontal:
        counts.plot.barh(ax=ax)
        ax.invert_yaxis()
        ax.set_xlabel("Count")
    else:
        counts.plot.bar(ax=ax)
        ax.set_ylabel("Count")

    ax.set_title(title)

    if save_name:
        save_fig(fig, save_name)
    return fig


def plot_distribution(
    series: pd.Series,
    title: str,
    xlabel: str,
    bins: int = 50,
    clip: Optional[tuple] = None,
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Histogram with KDE for a numeric Series."""
    data = series.dropna()
    if clip:
        data = data.clip(*clip)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    sns.histplot(data, bins=bins, kde=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")

    if save_name:
        save_fig(fig, save_name)
    return fig


def plot_box_by_category(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    order: Optional[list] = None,
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Box plot of a numeric column grouped by a categorical column."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    sns.boxplot(data=df, x=x, y=y, order=order, ax=ax)
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")

    if save_name:
        save_fig(fig, save_name)
    return fig


def plot_time_series(
    series: pd.Series,
    title: str,
    ylabel: str,
    freq: str = "W",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Line plot of counts over time, resampled to a given frequency."""
    counts = series.dropna().dt.to_period(freq).value_counts().sort_index()
    counts.index = counts.index.to_timestamp()

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(counts.index, counts.values, linewidth=1.5)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Date")
    fig.autofmt_xdate()

    if save_name:
        save_fig(fig, save_name)
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    title: str = "Correlation Heatmap",
    save_name: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of pairwise correlations for numeric columns."""
    corr = df.select_dtypes(include="number").corr()

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        ax=ax,
    )
    ax.set_title(title)

    if save_name:
        save_fig(fig, save_name)
    return fig

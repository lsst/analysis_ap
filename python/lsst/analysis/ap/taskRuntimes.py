# This file is part of analysis_ap.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Collect per-quantum runtimes from the ``*_metadata`` datasets of a
butler run collection.

The single public entry point, `collect_task_runtimes`, walks every
``<task>_metadata`` dataset, pulls the timing fields via
`lsst.pipe.base.resource_usage.QuantumResourceUsage.from_task_metadata`,
applies a per-task threshold (any-quantum-over-threshold keeps the
whole task), and returns a tidy DataFrame, optionally with a box plot.
"""

from __future__ import annotations

__all__ = ["collect_task_runtimes"]

import pandas as pd

from lsst.pipe.base.resource_usage import QuantumResourceUsage


def collect_task_runtimes(butler, collections, threshold=1.0, *,
                          plot=False, ax=None):
    """Per-task runtime and memory summary for a butler run collection.

    Each ``<task>_metadata`` dataset under ``collections`` is loaded and
    its timing fields extracted with
    `~lsst.pipe.base.resource_usage.QuantumResourceUsage.from_task_metadata`.
    Tasks whose every quantum runs faster than ``threshold`` seconds are
    dropped; tasks with at least one quantum at or above ``threshold``
    contribute all their quanta to the per-task summary statistics, so the
    summary reflects cross-quantum variability rather than a single outlier.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler used to query the registry and load metadata datasets.
    collections : `str` or iterable of `str`
        Collections to query, typically a single run collection name.
    threshold : `float`, optional
        Minimum task duration (seconds) for inclusion. A task is kept iff
        at least one of its quanta has ``total_time`` >= ``threshold``.
        Default ``1.0``.
    plot : `bool`, optional
        If True, render a horizontal box plot of per-quantum
        ``total_time`` per surviving task (tasks ordered by max
        ``total_time`` descending) and return the ``(df, fig)`` pair
        instead of just ``df``.
    ax : `matplotlib.axes.Axes` or None
        Axes to plot onto. Only used when ``plot=True``. If None, a new
        figure and axes are created.

    Returns
    -------
    df : `pandas.DataFrame`
        One row per task, with columns:

        - ``task``: pipeline task label
        - ``n_quanta``: number of surviving quanta contributing to the row
        - ``total_time_mean``, ``total_time_min``, ``total_time_max``,
          ``total_time_std``: seconds
        - ``memory_mean_<UNIT>``, ``memory_min_<UNIT>``,
          ``memory_max_<UNIT>``, ``memory_std_<UNIT>``: where ``UNIT`` is
          ``GB`` if any task's peak memory crosses 1 GB and ``MB``
          otherwise. The unit is chosen once across the whole table so
          the columns remain numerically comparable.
    fig : `matplotlib.figure.Figure`
        Only returned when ``plot=True``.
    """
    rows = []
    for dataset_type in butler.registry.queryDatasetTypes("*_metadata"):
        task = dataset_type.name[:-len("_metadata")]
        for ref in butler.registry.queryDatasets(dataset_type, collections=collections):
            metadata = butler.get(ref)
            try:
                usage = QuantumResourceUsage.from_task_metadata(metadata)
            except KeyError:
                # Quantum block exists but is missing one of the expected
                # fields (e.g. an aborted run); skip rather than fail.
                continue
            if usage is None:
                continue
            rows.append({"task": task,
                         "total_time": usage.total_time,
                         "memory": usage.memory})

    if not rows:
        return (pd.DataFrame(), None) if plot else pd.DataFrame()

    per_quantum = pd.DataFrame(rows)
    per_task_max = per_quantum.groupby("task")["total_time"].max()
    keep_tasks = per_task_max[per_task_max >= threshold].index
    per_quantum = per_quantum[per_quantum["task"].isin(keep_tasks)]

    summary = (per_quantum.groupby("task")
               .agg(n_quanta=("total_time", "count"),
                    total_time_mean=("total_time", "mean"),
                    total_time_min=("total_time", "min"),
                    total_time_max=("total_time", "max"),
                    total_time_std=("total_time", "std"),
                    memory_mean=("memory", "mean"),
                    memory_min=("memory", "min"),
                    memory_max=("memory", "max"),
                    memory_std=("memory", "std"))
               .reset_index()
               .sort_values("total_time_max", ascending=False)
               .reset_index(drop=True))

    mem_cols = ["memory_mean", "memory_min", "memory_max", "memory_std"]
    # Use GB if any task's peak memory crosses 1 GB, otherwise MB.
    if summary["memory_max"].max() >= 1024**3:
        mem_unit, mem_divisor = "GB", 1024**3
    else:
        mem_unit, mem_divisor = "MB", 1024**2
    summary[mem_cols] = summary[mem_cols] / mem_divisor
    summary = summary.rename(columns={c: f"{c}_{mem_unit}" for c in mem_cols})

    if not plot:
        return summary

    import matplotlib.pyplot as plt

    # Bottom-up order so the slowest task sits at the top of the horizontal
    # plot.
    order = per_task_max.loc[keep_tasks].sort_values(ascending=True).index.tolist()
    data = [per_quantum.loc[per_quantum["task"] == t, "total_time"].to_numpy() for t in order]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(order))))
    else:
        fig = ax.figure
    ax.boxplot(data, vert=False, tick_labels=order, showfliers=True)
    ax.set_xlabel("total_time (s)")
    if per_quantum["total_time"].max() > 100 * threshold:
        ax.set_xscale("log")
    ax.axvline(threshold, color="grey", linestyle=":", linewidth=1,
               label=f"threshold = {threshold:g} s")
    ax.set_title("Per-quantum total_time")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize="small")
    fig.tight_layout()
    return summary, fig

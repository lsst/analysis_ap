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

"""Visualization helpers for AP analysis.

Three tools live here:

- `lightcurve` plots a per-band psfFlux light curve for a single diaObject,
  optionally overlaying forced photometry. It works with either the APDB
  `DbQuery` interface (pandas) or the PPDB `PpdbTap` interface (astropy
  Tables).
- `cutout_grid` lays out science/template/difference cutouts for many
  DiaSources in a single mosaic figure.
- `summarize_run` returns a per-visit summary DataFrame of an APDB run
  (counts, dipole rate, reliability statistics, etc.) suitable for a quick
  health check of a processing run.
"""

from __future__ import annotations

__all__ = ["lightcurve", "cutout_grid", "summarize_run", "BAND_COLORS", "band_color"]

import inspect
import io

import numpy as np
import pandas as pd

from lsst.utils.plotting import get_multiband_plot_colors

# The official Rubin band colors (RTN-045 colorblind-friendly palette),
# keyed by lower-case band character. Use `band_color` rather than
# indexing this directly, so that unknown bands get a sensible fallback.
BAND_COLORS = get_multiband_plot_colors()


def band_color(band, default="k"):
    """Return the official Rubin plotting color for a band.

    Parameters
    ----------
    band : `str`
        Band name, e.g. ``"g"``.
    default : `str`, optional
        Color to return for a band with no official color.

    Returns
    -------
    color : `str`
        Matplotlib color specification.
    """
    return BAND_COLORS.get(band, default)


def _time_column(frame):
    """Return the name of the MJD-like time column on a DiaSource frame.

    Older APDB schemas used ``midPointTai``; current ones use
    ``midpointMjdTai``. This helper accepts either.
    """
    for candidate in ("midpointMjdTai", "midPointTai"):
        if candidate in frame.columns:
            return candidate
    raise KeyError("Expected one of 'midpointMjdTai' or 'midPointTai' "
                   f"in DataFrame; got columns: {list(frame.columns)}")


def _to_dataframe(table):
    """Return ``table`` as a `pandas.DataFrame`.

    The APDB `DbQuery` loaders already return DataFrames; the PPDB `PpdbTap`
    loaders return `astropy.table.Table`. Normalizing here lets the plotting
    code use a single pandas (groupby-based) path regardless of which
    interface produced the data. Masked astropy values become NaN.
    """
    if isinstance(table, pd.DataFrame):
        return table
    return table.to_pandas()


def _load_object_sources(query, dia_object_id, exclude_flagged):
    """Load one diaObject's DiaSources as a DataFrame across query interfaces.
    """
    method = query.load_sources_for_object
    if "exclude_flagged" in inspect.signature(method).parameters:
        sources = method(dia_object_id, exclude_flagged=exclude_flagged)
    elif exclude_flagged:
        raise TypeError(
            f"{type(query).__name__}.load_sources_for_object does not support "
            "exclude_flagged; the PPDB public interface does not expose "
            "diaSource flag filtering. Pass exclude_flagged=False.")
    else:
        sources = method(dia_object_id)
    return _to_dataframe(sources)


def lightcurve(query, dia_object_id, ax=None, exclude_flagged=False,
               include_forced=True):
    """Plot a per-band psfFlux light curve for one diaObject.

    Parameters
    ----------
    query : `lsst.analysis.ap.apdb.DbQuery` or \
            `lsst.analysis.ap.ppdb.PpdbTap`
    dia_object_id : `int`
        Object id to load.
    ax : `matplotlib.axes.Axes`, optional
        Axes to draw into; if None, a new figure is created.
    exclude_flagged : `bool`, optional
        Forwarded to `load_sources_for_object` when the query supports it.
        Defaults to False so the lightcurve matches the row count of a direct
        APDB query; pass True to drop diaSources matching the configured
        bad-flag list. The PPDB `PpdbTap` interface does not expose flag
        filtering, so True is rejected there. DiaForcedSources are always
        loaded unfiltered.
    include_forced : `bool`, optional
        If True, also overlay diaForcedSources as small markers.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
    ax : `matplotlib.axes.Axes`
    sources : `pandas.DataFrame`
        DiaSources used for the plot.
    forced : `pandas.DataFrame` or None
        DiaForcedSources used for the plot (None if ``include_forced`` is
        False).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    # diaObjectId is NaN for diaSources not associated with any diaObject;
    # short-circuit before hitting the database.
    if pd.isna(dia_object_id):
        ax.text(0.5, 0.5, "no diaObjectId (NaN)",
                ha="center", va="center", transform=ax.transAxes)
        return fig, ax, pd.DataFrame(), None

    sources = _load_object_sources(query, dia_object_id, exclude_flagged)
    # DiaForcedSource has a different (and smaller) flag schema than
    # DiaSource: applying the diaSource exclusion list would key into
    # columns that don't exist on the forced table. Forced photometry is
    # also a measurement at a known location rather than a fresh detection,
    # so showing it unfiltered is the right behavior.
    forced = (_to_dataframe(query.load_forced_sources_for_object(dia_object_id))
              if include_forced else None)

    if len(sources) == 0:
        ax.text(0.5, 0.5, f"no sources for diaObjectId={dia_object_id}",
                ha="center", va="center", transform=ax.transAxes)
        return fig, ax, sources, forced

    time_col = _time_column(sources)
    for band, group in sources.groupby("band"):
        color = band_color(band)
        ax.errorbar(group[time_col], group["psfFlux"], yerr=group["psfFluxErr"],
                    fmt="o", color=color, label=f"{band} (n={len(group)})")

    if forced is not None and len(forced):
        forced_time_col = _time_column(forced)
        # Plot per-band forced points in their band colors, but suppress
        # individual legend entries so the forced marker is represented
        # once (in black) regardless of how many bands are present.
        for band, group in forced.groupby("band"):
            color = band_color(band)
            ax.errorbar(group[forced_time_col], group["psfFlux"],
                        yerr=group["psfFluxErr"], fmt=".", ms=4, color=color,
                        alpha=0.4, label="_nolegend_")
        ax.plot([], [], ".", color="black", ms=4, alpha=0.4,
                label=f"forced (n={len(forced)})")

    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel(time_col)
    ax.set_ylabel("psfFlux (nJy)")
    ax.set_title(f"diaObjectId = {dia_object_id}")
    ax.legend(frameon=True)
    return fig, ax, sources, forced


def cutout_grid(sources, butler, instrument, n_per_row=4, config=None, output=None,
                figsize=None, ra_column='ra', dec_column='dec', detector_column='detector',
                visit_column='visit', id_column='diaSourceId'):
    """Render science/template/difference cutouts for many sources in a grid.

    This is a thin wrapper around `PlotImageSubtractionCutoutsTask`: it calls
    `generate_image` for each source (which returns a PNG in memory) and
    arranges the resulting rasters in a single matplotlib figure.

    Parameters
    ----------
    sources : `pandas.DataFrame`
        DiaSources to cut out. Must contain at least
        ``ra, dec, diaSourceId, detector, visit, instrument`` plus whatever
        annotation fields the task config requires (see
        ``PlotImageSubtractionCutoutsConfig.add_metadata``).
    butler : `lsst.daf.butler.Butler`
        Butler initialized with the relevant collections.
    instrument : `str`
        Name of the instrument for the data being plotted.
    n_per_row : `int`
        Number of cutouts per row in the resulting figure.
    config : `PlotImageSubtractionCutoutsConfig`, optional
        Cutout config to use (see
        ``plotImageSubtractionCutouts.PlotImageSubtractionCutoutsConfig``).
        Defaults to a fresh instance with ``add_metadata=False``
        (annotations get cluttered in a grid).
    output : `str`, optional
        If given, save the figure to this path with ``bbox_inches="tight"``.
    figsize : `tuple` [`float`, `float`], optional
        Figure size in inches. Defaults to ``(n_per_row*3.5, n_rows*1.7)``.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
    """
    import matplotlib.pyplot as plt
    import PIL.Image

    import lsst.geom

    # Local import to avoid a circular dependency at module load time.
    from . import plotImageSubtractionCutouts as cutouts_mod

    if config is None:
        config = cutouts_mod.PlotImageSubtractionCutoutsConfig()
        # Annotations get cramped in a grid; default them off here.
        config.add_metadata = False

    task = cutouts_mod.PlotImageSubtractionCutoutsTask(config=config, output_path="")
    cutouts_mod.butler_cache.set(butler, config)

    n_sources = len(sources)
    n_rows = max(1, (n_sources + n_per_row - 1) // n_per_row)
    if figsize is None:
        figsize = (n_per_row * 3.5, n_rows * 1.7)
    fig, axes = plt.subplots(n_rows, n_per_row, figsize=figsize, squeeze=False)

    records = sources.to_records(index=False)
    for i, source in enumerate(records):
        row, col = divmod(i, n_per_row)
        ax = axes[row][col]
        ax.set_axis_off()
        try:
            sci, tmpl, diff = cutouts_mod.butler_cache.get_exposures(
                instrument, source[detector_column], source[visit_column])
            center = lsst.geom.SpherePoint(source[ra_column], source[dec_column],
                                           lsst.geom.degrees)
            scale = sci.wcs.getPixelScale(sci.getBBox().getCenter()).asArcseconds()
            png = task.generate_image(
                sci, tmpl, diff, center, scale,
                source=source if config.add_metadata else None,
            )
            with PIL.Image.open(io.BytesIO(png.getvalue())) as img:
                ax.imshow(np.asarray(img))
            ax.set_title(f"{source[id_column]}", fontsize=7)
        except Exception as exc:
            ax.text(0.5, 0.5, f"{type(exc).__name__}", ha="center", va="center",
                    fontsize=8, transform=ax.transAxes)

    # Blank out any trailing axes in the last row.
    for i in range(n_sources, n_rows * n_per_row):
        row, col = divmod(i, n_per_row)
        axes[row][col].set_axis_off()

    fig.tight_layout()
    if output is not None:
        fig.savefig(output, bbox_inches="tight")
    # Remove the figure from pyplot's figure manager so the Jupyter inline
    # backend does not auto-display it at end-of-cell in addition to the
    # caller's own rendering of the returned Figure (which would duplicate
    # the output the first time the function is run in a notebook). The
    # returned Figure object remains valid and renders via its repr hooks.
    plt.close(fig)
    return fig


def summarize_run(query, bad_flag_list=None):
    """Return a per-visit summary DataFrame for an APDB run.

    Useful as a quick health check after a processing run: one row per visit
    with source counts, dipole rate, reliability statistics, and the fraction
    of sources that fall in the bad-flag list.

    Parameters
    ----------
    query : `lsst.analysis.ap.apdb.DbQuery`
        APDB query interface (sqlite, postgres, or cassandra).
    bad_flag_list : `list` [`str`], optional
        Flag column names to count as "bad". If omitted, the query's
        currently-configured exclusion list is used. The caller's exclusion
        list is restored before returning.

    Returns
    -------
    summary : `pandas.DataFrame`
        Indexed by ``visit``. Columns:
        ``n_sources``, ``n_unflagged``, ``bad_flag_fraction``,
        ``median_reliability`` (if column present),
        ``dipole_fraction`` (if column present),
        ``median_psf_chi2_per_dof`` (if columns present).
    """
    saved_flags = list(query.diaSource_flags_exclude)
    try:
        if bad_flag_list is not None:
            query.set_excluded_diaSource_flags(bad_flag_list)
        sources_all = query.load_sources(limit=None)
        sources_clean = query.load_sources(exclude_flagged=True, limit=None)
    finally:
        query.set_excluded_diaSource_flags(saved_flags)

    if len(sources_all) == 0:
        return pd.DataFrame()

    clean_per_visit = sources_clean.groupby("visit").size() if len(sources_clean) else pd.Series(dtype=int)

    rows = []
    for visit, group in sources_all.groupby("visit"):
        n_all = len(group)
        n_clean = int(clean_per_visit.get(visit, 0))
        row = {
            "visit": visit,
            "n_sources": n_all,
            "n_unflagged": n_clean,
            "bad_flag_fraction": 1.0 - n_clean / n_all if n_all else 0.0,
        }
        if "reliability" in group.columns:
            row["median_reliability"] = group["reliability"].median()
        if "isDipole" in group.columns:
            row["dipole_fraction"] = float(group["isDipole"].mean())
        if "psfChi2" in group.columns and "psfNdata" in group.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = group["psfChi2"] / group["psfNdata"]
            row["median_psf_chi2_per_dof"] = float(np.nanmedian(ratio))
        rows.append(row)
    return pd.DataFrame(rows).set_index("visit").sort_index()

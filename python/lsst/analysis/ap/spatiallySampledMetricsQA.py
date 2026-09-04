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

"""Notebook-friendly QA for ``SpatiallySampledMetricsTask`` output.

The single public entry point `subtraction_quality_report` consumes a
``*_spatiallySampledMetrics`` table for one detector and produces:

- A printed top-line summary of the three diagnostic scalars
  (`diffim_chi2PerPix`, `psfMatchingKernel_residualNorm`,
  `dipole_density`) at percentiles that make localized failures visible.
- A three-panel diagnostic figure with the three scalars rendered as
  linear-interpolated heatmaps, with the kernel centroid offset quiver
  (colored cyclically by angle, magnitude shown by a per-panel
  quiverkey reference arrow) overlaid on each panel.

See the module-level constants for the metric reference values and the
default mask-fraction columns used to filter samples sitting on bad
detector regions before computing statistics.
"""

from __future__ import annotations

__all__ = ["subtraction_quality_report"]

import numpy as np
import pandas as pd

from lsst.analysis.ap.skymapOverlay import make_affine_sky_to_xy, draw_skymap_outlines_mpl

# Mask-fraction columns whose sum indicates a sample sits on an unusable
# region of the detector. The headline scalars are computed only on samples
# below ``bad_mask_threshold`` so the distribution tails reflect subtraction
# quality rather than edge / saturated pixels.
DEFAULT_BAD_MASK_COLUMNS = (
    "bad_mask_fraction",
    "sat_mask_fraction",
    "edge_mask_fraction",
    "no_data_mask_fraction",
)

# Default extra padding (in detector pixels) added to each panel beyond
# the autoscaled data limits, so labels anchored at the edge of the data have
# room to render without being clipped by the panel boundary. NOTE: the
# displayed figure is at significantly lower resolution than the original
# image.
_DEFAULT_PANEL_PADDING_PIX = 150

# (column, reference value, display label). Reference is the value the
# metric takes on a perfectly subtracted, well-decorrelated diffim.
_HEADLINE_METRICS = (
    ("diffim_chi2PerPix",              1.0, "Diffim chi^2/pix"),  # noqa: E241
    ("psfMatchingKernel_residualNorm", 0.0, "PSF match residual"),  # noqa: E241
    ("dipole_density",                 0.0, "Dipoles / deg^2"),  # noqa: E241
)


def _coerce_to_frame(metrics):
    """Accept an astropy Table, DataFrame, or any DataFrame-castable input."""
    if isinstance(metrics, pd.DataFrame):
        return metrics
    if hasattr(metrics, "to_pandas"):
        return metrics.to_pandas()
    return pd.DataFrame(metrics)


def _filter_clean(df, bad_mask_threshold, bad_mask_columns):
    """Drop samples whose summed bad-mask fractions exceed the threshold."""
    cols = [c for c in bad_mask_columns if c in df.columns]
    if not cols:
        return df.copy()
    return df[df[cols].sum(axis=1) < bad_mask_threshold].copy()


def _percentile_row(values):
    """Return formatted (median, p84, p95, p99) strings for a 1-D array."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return ["nan"]*4
    q = np.percentile(finite, [50, 84, 95, 99])
    return [f"{x:.3f}" for x in q]


def _print_summary(clean, n_total, threshold):
    """Print a fixed-width summary of the three headline metrics."""
    headers = ["metric", "ref", "median", "p84", "p95", "p99"]
    rows = []
    for col, ref, _label in _HEADLINE_METRICS:
        if col not in clean.columns:
            rows.append([col, f"{ref:.2f}", "n/a", "n/a", "n/a", "n/a"])
            continue
        rows.append([col, f"{ref:.2f}", *_percentile_row(clean[col].to_numpy())])
    widths = [max(len(str(r[i])) for r in [headers, *rows]) for i in range(len(headers))]

    def _fmt(row):
        return "  ".join(str(c).ljust(w) for c, w in zip(row, widths))

    print(f"SpatiallySampledMetrics: {len(clean)}/{n_total} samples retained "
          f"(bad-mask fraction sum < {threshold:g})")
    print()
    print(_fmt(headers))
    print("  ".join("-"*w for w in widths))
    for row in rows:
        print(_fmt(row))


def _metric_panel(ax, fig, clean, col, vmin, vmax, label, cmap, vcenter=None,
                  panel_padding_pix=_DEFAULT_PANEL_PADDING_PIX):
    """Render one metric panel: linear-interpolated heatmap with sample
    markers.

    Points whose (x, y, value) tuple contains any NaN are dropped before
    interpolation. Grid cells outside the convex hull of the surviving
    samples are left as NaN, which the colormap renders as transparent.

    If ``vcenter`` is provided and strictly between ``vmin`` and ``vmax``,
    the panel uses a ``TwoSlopeNorm`` so the colormap midpoint always
    maps to ``vcenter`` regardless of whether the (vmin, vmax) range is
    symmetric about it.
    """
    if col not in clean.columns or not clean[col].notna().any():
        ax.text(0.5, 0.5, f"{col}\n(not present)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(col)
        return

    x = clean["x"].to_numpy()
    y = clean["y"].to_numpy()
    z = clean[col].to_numpy()
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if not valid.any():
        ax.text(0.5, 0.5, f"{col}\n(no valid samples)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(col)
        return
    xv, yv, zv = x[valid], y[valid], z[valid]

    # vmin and vmax autoscale independently: an explicit ``vmin=0.0`` from
    # the caller is never overridden by the vmax-None branch.
    if vmin is None:
        vmin = 0.0
    if vmax is None:
        vmax = float(np.nanpercentile(zv, 99))
    # Set a reasonable colorbar scale even if the data is completely constant.
    if not (vmax > vmin):
        vmax = vmin + 1.0

    from scipy.interpolate import griddata
    from matplotlib.colors import Normalize, TwoSlopeNorm
    grid_size = 200
    xi = np.linspace(xv.min(), xv.max(), grid_size)
    yi = np.linspace(yv.min(), yv.max(), grid_size)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata((xv, yv), zv, (XI, YI), method="linear")

    # Center the colormap on ``vcenter`` when supplied and well-posed.
    # TwoSlopeNorm requires vmin < vcenter < vmax strictly; if the
    # autoscaled range collapses around vcenter we fall back to a plain
    # clipping Normalize rather than raise.
    if vcenter is not None and vmin < vcenter < vmax:
        norm = TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    im = ax.imshow(ZI, origin="lower",
                   extent=(xv.min(), xv.max(), yv.min(), yv.max()),
                   norm=norm, cmap=cmap, aspect="equal")
    fig.colorbar(im, ax=ax, label=label)
    # Overlay the kernel centroid quiver so direction information is
    # available next to every metric heatmap, with a quiverkey reference
    # arrow. Sample markers go on top with higher "zorder" so the positions
    # stay visible through arrows.
    quiver_info = _overlay_kernel_quiver(ax, clean)
    if quiver_info is not None:
        q, ref_length = quiver_info
        ax.quiverkey(q, 0.85, 1.05, ref_length, f"{ref_length:.2f}″",
                     labelpos="E", coordinates="axes")
    ax.scatter(xv, yv, s=8, facecolors="white", edgecolors="black",
               linewidths=0.3, zorder=5)
    ax.set_title(col)
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    ax.set_aspect("equal")
    # Pad the data limits relative to whatever the artists left them at,
    # so labels (and quiver heads) sitting at the very edge of the data
    # are not clipped by the panel boundary.
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.set_xlim(xmin - panel_padding_pix, xmax + panel_padding_pix)
    ax.set_ylim(ymin - panel_padding_pix, ymax + panel_padding_pix)


def _overlay_kernel_quiver(ax, clean):
    """Draw the HSV-colored kernel centroid quiver onto ``ax``.

    Returns
    -------
    info : tuple or None
        ``(quiver_artist, ref_length_arcsec)`` on success, or None if the
        required columns are missing or no samples are valid. The caller
        decides whether to render a colorbar / quiverkey for it.
    """
    needed = ("psfMatchingKernel_length", "psfMatchingKernel_direction", "x", "y")
    if not all(c in clean.columns for c in needed):
        return None
    length = clean["psfMatchingKernel_length"].to_numpy()
    direction = clean["psfMatchingKernel_direction"].to_numpy()
    u = length*np.cos(direction)
    v = length*np.sin(direction)
    ok = np.isfinite(u) & np.isfinite(v) & (length > 0)
    if not ok.any():
        return None

    ref_length = float(np.nanpercentile(length[ok], 95))
    direction_deg = np.degrees(direction[ok]) % 360.0
    x = clean["x"].to_numpy()[ok]
    y = clean["y"].to_numpy()[ok]
    q = ax.quiver(x, y, u[ok], v[ok], direction_deg,
                  cmap="hsv", clim=(0, 360),
                  angles="xy", scale_units="xy",
                  scale=ref_length/200, width=0.004, pivot="mid",
                  zorder=4)
    return q, ref_length


def _draw_skymap_outlines(ax, skymap, clean, label_fontsize=7):
    """Overlay patch boundaries (with tract,patch labels) on a panel.

    The metrics table carries no WCS, so the sky↔detector pixel mapping is
    derived from the sample positions' ``(x, y)`` and
    ``(coord_ra, coord_dec)`` columns via a least-squares affine fit (see
    `~lsst.analysis.ap.skymapOverlay.make_affine_sky_to_xy`). For a single
    detector this is typically accurate to well under a pixel — enough for
    visualization but not for science.

    Silently no-ops when sky coordinates aren't available or fewer than
    three samples are valid.
    """
    if "coord_ra" not in clean.columns or "coord_dec" not in clean.columns:
        return

    ra = clean["coord_ra"].to_numpy()
    dec = clean["coord_dec"].to_numpy()
    xs = clean["x"].to_numpy()
    ys = clean["y"].to_numpy()
    valid = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(xs) & np.isfinite(ys)
    if valid.sum() < 3:
        return

    import lsst.geom as geom

    sky_to_xy = make_affine_sky_to_xy(ra[valid], dec[valid], xs[valid], ys[valid])

    # Build a coord list spanning the sample footprint so findTractPatchList
    # returns every tract / patch that touches it.
    corner_pairs = [
        (ra[valid].min(), dec[valid].min()),
        (ra[valid].max(), dec[valid].min()),
        (ra[valid].max(), dec[valid].max()),
        (ra[valid].min(), dec[valid].max()),
    ]
    sky_corners = [geom.SpherePoint(r, d, geom.radians) for r, d in corner_pairs]

    draw_skymap_outlines_mpl(ax, skymap, sky_to_xy, sky_corners,
                             label_fontsize=label_fontsize)


def _make_figure(clean, chi2_scale=1.0, skymap=None, label_fontsize=7,
                 panel_padding_pix=_DEFAULT_PANEL_PADDING_PIX):
    """Build a three-panel diagnostic figure.

    Parameters
    ----------
    clean : `pandas.DataFrame`
        The filtered metrics table.
    chi2_scale : `float`, optional
        Half-width of the ``diffim_chi2PerPix`` colormap range, measured
        multiplicatively about the nominal value of 1.0. ``vmax`` is set
        to ``1 + chi2_scale`` and ``vmin`` to its reciprocal so the
        colorbar covers the same factor above and below nominal in log space.
    skymap : `lsst.skymap.BaseSkyMap`, optional
        If supplied, overlay patch outlines (with tract,patch labels) on
        every panel.
    label_fontsize : `int` or `float`, optional
        Font size for the per-patch ``tract,patch`` labels.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # White-to-black sequential colormap for dipole density: pure white
    # at the floor, pure black at vmax. NaN cells (outside the
    # interpolation convex hull) are mapped to ``lightblue`` instead of
    # any grey because every grey lives somewhere inside the white→black
    # gradient and would otherwise read as a real mid-range value.
    white_to_black = LinearSegmentedColormap.from_list(
        "white_to_black", [(1.0, 1.0, 1.0), (0.0, 0.0, 0.0)]).copy()
    white_to_black.set_bad("lightblue")

    chi2_vmax = 1.0 + chi2_scale
    chi2_vmin = 1.0/chi2_vmax

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    # (col, vmin, vmax, label, cmap, vcenter). ``vcenter`` is non-None only
    # for panels whose colormap should be anchored at a known reference
    # value -- e.g. chi^2/pix is centered on its nominal value of 1.0.
    panels = (
        ("diffim_chi2PerPix",              chi2_vmin, chi2_vmax, "Diffim chi^2/pix",   "RdBu_r",       1.0),  # noqa: E241,E501
        ("psfMatchingKernel_residualNorm", 0.0,       0.2,       "PSF match residual", "viridis",      None),  # noqa: E241,E501
        ("dipole_density",                 0.0,       None,      "Dipoles / deg^2",    white_to_black, None),  # noqa: E241,E501
    )
    for ax, (col, vmin, vmax, label, cmap, vcenter) in zip(axes, panels):
        _metric_panel(ax, fig, clean, col, vmin, vmax, label, cmap, vcenter=vcenter,
                      panel_padding_pix=panel_padding_pix)
        if skymap is not None:
            _draw_skymap_outlines(ax, skymap, clean, label_fontsize=label_fontsize)
    return fig


def subtraction_quality_report(metrics,
                               bad_mask_threshold=0.2,
                               bad_mask_columns=DEFAULT_BAD_MASK_COLUMNS,
                               chi2_scale=1.0,
                               skymap=None,
                               label_fontsize=7,
                               panel_padding_pix=_DEFAULT_PANEL_PADDING_PIX):
    """Print a headline metric summary and build the diagnostic plot.

    Parameters
    ----------
    metrics : `astropy.table.Table` or `pandas.DataFrame`
        The metrics table produced by ``SpatiallySampledMetricsTask`` for a
        single detector.
    bad_mask_threshold : `float`, optional
        Samples whose summed mask fractions across ``bad_mask_columns``
        meet or exceed this value are dropped before computing statistics
        and rendering the figure.
    bad_mask_columns : iterable of `str`, optional
        Names of mask-fraction columns to sum when deciding whether a
        sample sits on a usable patch of the detector. Columns missing
        from the input table are silently ignored.
    chi2_scale : `float`, optional
        Multiplicative half-width of the ``diffim_chi2PerPix`` colormap
        around its nominal value of 1.0. The colorbar covers
        ``[1/(1+chi2_scale), 1+chi2_scale]`` so a deviation by a factor
        of ``(1+chi2_scale)`` above or below 1 sits at the colormap extremes.
    skymap : `lsst.skymap.BaseSkyMap`, optional
        If supplied, overlay the boundaries of every patch that touches
        the detector footprint on each panel, with a ``tract,patch``
        label anchored just inside the lower-left corner of each patch.
        The local sky↔pixel mapping is inferred from the sample
        positions' ``(x, y)`` and ``(coord_ra, coord_dec)`` columns via
        a least-squares affine fit, so the alignment is approximate
        When omitted, no outlines are drawn.
    label_fontsize : `int` or `float`, optional
        Font size for the per-patch ``tract,patch`` labels. Only used
        when ``skymap`` is supplied. Default 7.
    panel_padding_pix : `float`, optional
        Detector-pixel buffer added on each side of every panel beyond
        the autoscaled data limits.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        The 1x3 diagnostic figure (`diffim_chi2PerPix`, PSF match
        residual, dipole density). The kernel centroid offset quiver is
        overlaid on every panel, with a quiverkey reference arrow.

    Notes
    -----
    The headline summary is printed via ``print``, so the function is
    intended for direct use in a notebook cell. Pass the returned figure
    to ``fig.savefig(...)`` if you want to persist the diagnostic.
    """
    df = _coerce_to_frame(metrics)
    clean = _filter_clean(df, bad_mask_threshold, bad_mask_columns)
    _print_summary(clean, n_total=len(df), threshold=bad_mask_threshold)
    return _make_figure(clean, chi2_scale=chi2_scale, skymap=skymap,
                        label_fontsize=label_fontsize,
                        panel_padding_pix=panel_padding_pix)

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

    The sky↔detector pixel mapping is derived from the sample positions'
    ``(x, y)`` and ``(coord_ra, coord_dec)`` columns via a least-squares
    affine fit. For a single detector this is typically accurate to well
    under a pixel — enough for visualization but not for science.

    Each overlapping patch gets a thin outline plus a ``tract,patch``
    label placed along the midpoint of its longest visible edge inside
    the panel.

    When more than one tract overlaps the panel (e.g. detectors near a
    tract boundary), patches are distinguished by linestyle. Tracts are
    ranked by their total visible patch area within the panel, and
    assigned: solid, dashed, dotted, dash-dotted.

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
    import matplotlib.patheffects as pe

    # Preserve the current view: tract outlines almost always extend far
    # beyond the detector footprint, and matplotlib would otherwise auto-
    # rescale the panel to fit them, shrinking the actual data to a dot.
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    A = np.column_stack([np.ones(valid.sum()), ra[valid], dec[valid]])
    coef_x, *_ = np.linalg.lstsq(A, xs[valid], rcond=None)
    coef_y, *_ = np.linalg.lstsq(A, ys[valid], rcond=None)

    def sky_to_xy(sphere_point):
        r = sphere_point.getRa().asRadians()
        d = sphere_point.getDec().asRadians()
        return (float(coef_x[0] + coef_x[1]*r + coef_x[2]*d),
                float(coef_y[0] + coef_y[1]*r + coef_y[2]*d))

    def _draw_polygon(corners_tract_px, tract_wcs, **plot_kwargs):
        sky_corners = [tract_wcs.pixelToSky(p) for p in corners_tract_px]
        xy = [sky_to_xy(c) for c in sky_corners]
        xs_c = [p[0] for p in xy] + [xy[0][0]]
        ys_c = [p[1] for p in xy] + [xy[0][1]]
        ax.plot(xs_c, ys_c, **plot_kwargs)

    # Build a coord list spanning the sample footprint so findTractPatchList
    # returns every tract / patch that touches it.
    corner_pairs = [
        (ra[valid].min(), dec[valid].min()),
        (ra[valid].max(), dec[valid].min()),
        (ra[valid].max(), dec[valid].max()),
        (ra[valid].min(), dec[valid].max()),
    ]
    coord_list = [geom.SpherePoint(r, d, geom.radians) for r, d in corner_pairs]
    tract_patch_list = skymap.findTractPatchList(coord_list)

    # White lines with a thin black stroke read on any colormap background.
    line_outline = [pe.withStroke(linewidth=2.0, foreground="black")]
    text_outline = [pe.withStroke(linewidth=1.4, foreground="black")]

    xmin, xmax = xlim
    ymin, ymax = ylim

    # Pre-project every patch's corners into panel pixel coords, and
    # accumulate the visible patch area for each tract so we can rank
    # tracts by how much of the image they cover.
    tract_entries = []  # list of (visible_area, tract_info, [(patch, xy_corners), ...])
    for tract_info, patches in tract_patch_list:
        tract_wcs = tract_info.wcs
        per_tract_patches = []
        per_tract_area = 0.0
        for patch in patches:
            bbox = patch.getInnerBBox()
            corners_px = [geom.Point2D(bbox.minX, bbox.minY),
                          geom.Point2D(bbox.maxX, bbox.minY),
                          geom.Point2D(bbox.maxX, bbox.maxY),
                          geom.Point2D(bbox.minX, bbox.maxY)]
            xy_corners = [sky_to_xy(tract_wcs.pixelToSky(p))
                          for p in corners_px]
            clipped = _clip_polygon_to_rect(xy_corners,
                                            xmin, xmax, ymin, ymax)
            per_tract_area += _polygon_area(clipped)
            per_tract_patches.append((patch, corners_px, xy_corners))
        tract_entries.append((per_tract_area, tract_info, per_tract_patches))

    # Linestyles for the four most-overlapping tracts, in rank order.
    linestyles = ("-", "--", ":", "-.")
    # Sort by descending overlap area. Ties are broken by tract id so the
    # ordering is deterministic from one call to the next on the same
    # dataset.
    tract_entries.sort(key=lambda e: (-e[0], e[1].getId()))

    for rank, (area, tract_info, per_tract_patches) in enumerate(tract_entries):
        tract_wcs = tract_info.wcs
        tract_id = tract_info.getId()
        linestyle = linestyles[rank] if rank < len(linestyles) else "-"
        patch_kwargs = dict(color="white", linewidth=0.5, alpha=0.85,
                            linestyle=linestyle,
                            path_effects=line_outline, zorder=6)
        for patch, corners_px, xy_corners in per_tract_patches:
            bbox = patch.getInnerBBox()
            _draw_polygon(corners_px, tract_wcs, **patch_kwargs)

            # Place the label at the midpoint of the patch edge with the
            # longest visible portion within the panel, offset slightly
            # toward the patch center so the text sits inside.
            center_tract = geom.Point2D(0.5*(bbox.minX + bbox.maxX),
                                        0.5*(bbox.minY + bbox.maxY))
            cx, cy = sky_to_xy(tract_wcs.pixelToSky(center_tract))

            best = None
            for i in range(4):
                (x0, y0), (x1, y1) = xy_corners[i], xy_corners[(i + 1) % 4]
                clipped = _clip_segment_to_rect(x0, y0, x1, y1,
                                                xmin, xmax, ymin, ymax)
                if clipped is None:
                    continue
                cx0, cy0, cx1, cy1 = clipped
                length = float(np.hypot(cx1 - cx0, cy1 - cy0))
                if best is None or length > best[0]:
                    best = (length, cx0, cy0, cx1, cy1)
            if best is None:
                continue  # entire patch is outside the panel
            length, cx0, cy0, cx1, cy1 = best
            mx = 0.5*(cx0 + cx1)
            my = 0.5*(cy0 + cy1)
            # Inward offset toward the projected patch center. Use the
            # smaller of "fixed fraction of edge length" and "fraction of
            # the midpoint-to-center distance" so the offset never lands
            # outside the patch on slivers.
            dx, dy = cx - mx, cy - my
            d_center = float(np.hypot(dx, dy))
            if d_center > 0:
                step = min(0.06*length, 0.4*d_center)
                mx += dx/d_center * step
                my += dy/d_center * step

            # Rotate the text to lie parallel to the visible edge, flipping
            # to keep it reading right-side-up (angle clamped to [-90, 90]).
            angle_deg = float(np.degrees(np.arctan2(cy1 - cy0, cx1 - cx0)))
            if angle_deg > 90.0:
                angle_deg -= 180.0
            elif angle_deg < -90.0:
                angle_deg += 180.0

            ax.text(mx, my,
                    f"{tract_id},{patch.getSequentialIndex()}",
                    ha="center", va="center",
                    rotation=angle_deg, rotation_mode="anchor",
                    color="white", fontsize=label_fontsize,
                    path_effects=text_outline, zorder=7,
                    clip_on=True)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def _clip_polygon_to_rect(polygon, xmin, xmax, ymin, ymax):
    """Clip a convex polygon against an axis-aligned rectangle.

    Parameters
    ----------
    polygon : sequence of ``(x, y)`` tuples
        Vertices of the (convex) input polygon, in order.
    xmin, xmax, ymin, ymax : `float`
        The clipping rectangle.

    Returns
    -------
    clipped : `list` of ``(x, y)`` tuples
        The clipped polygon, or an empty list if the polygon lies
        entirely outside the rectangle.
    """
    # Each clip edge is parameterized by ("axis", value, keep_side)
    # where keep_side is +1 if "inside" means coordinate >= value,
    # -1 if "inside" means coordinate <= value.
    edges = (("x", xmin, +1), ("x", xmax, -1),
             ("y", ymin, +1), ("y", ymax, -1))

    def _inside(point, axis, val, sign):
        coord = point[0] if axis == "x" else point[1]
        return (coord - val)*sign >= 0.0

    def _intersect(p1, p2, axis, val):
        x1, y1 = p1
        x2, y2 = p2
        if axis == "x":
            t = (val - x1)/(x2 - x1)
            return (val, y1 + t*(y2 - y1))
        t = (val - y1)/(y2 - y1)
        return (x1 + t*(x2 - x1), val)

    output = list(polygon)
    for axis, val, sign in edges:
        if not output:
            return []
        input_list = output
        output = []
        for i in range(len(input_list)):
            curr = input_list[i]
            prev = input_list[i - 1]
            curr_in = _inside(curr, axis, val, sign)
            prev_in = _inside(prev, axis, val, sign)
            if curr_in:
                if not prev_in:
                    output.append(_intersect(prev, curr, axis, val))
                output.append(curr)
            elif prev_in:
                output.append(_intersect(prev, curr, axis, val))
    return output


def _polygon_area(polygon):
    """Area of a polygon via the shoelace formula."""
    n = len(polygon)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        s += x1*y2 - x2*y1
    return abs(s)*0.5


def _clip_segment_to_rect(x0, y0, x1, y1, xmin, xmax, ymin, ymax):
    """Liang-Barsky line-segment clipping against an axis-aligned rect.

    Returns the clipped endpoints ``(x0', y0', x1', y1')`` or ``None`` if
    the segment lies entirely outside the rectangle.
    """
    dx = x1 - x0
    dy = y1 - y0
    p = (-dx, dx, -dy, dy)
    q = (x0 - xmin, xmax - x0, y0 - ymin, ymax - y0)
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0.0:
            if qi < 0.0:
                return None
        else:
            t = qi/pi
            if pi < 0.0:
                if t > u2:
                    return None
                if t > u1:
                    u1 = t
            else:
                if t < u1:
                    return None
                if t < u2:
                    u2 = t
    return (x0 + u1*dx, y0 + u1*dy, x0 + u2*dx, y0 + u2*dy)


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

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

"""Backend-agnostic skymap tract/patch overlays.

The geometry of projecting overlapping tract/patch boundaries into a
display's pixel coordinate system is shared by two very different
renderers:

- `draw_skymap_outlines_mpl`, which draws onto a Matplotlib axis whose
  pixel<->sky relationship is only known approximately (e.g. the
  spatially-sampled-metrics panels, which carry sampled ``(x, y)`` and
  ``(coord_ra, coord_dec)`` but no WCS). Use `make_affine_sky_to_xy` to
  build the required ``sky_to_xy`` callable from those samples.
- `draw_skymap_outlines_afw`, which draws onto an `lsst.afw.display`
  frame showing an exposure with a real WCS, so the sky<->pixel mapping
  is exact.

Both renderers share `compute_tract_patch_outlines`, which does the
findTractPatchList lookup, projects every patch corner through a caller
supplied ``sky_to_xy`` map, and ranks the overlapping tracts by how much
of the display they cover.
"""

from __future__ import annotations

__all__ = ["make_affine_sky_to_xy", "compute_tract_patch_outlines",
           "draw_skymap_outlines_mpl", "draw_skymap_outlines_afw"]

import numpy as np


def make_affine_sky_to_xy(ra, dec, x, y):
    """Build a least-squares affine map from sky to detector pixels.

    Useful when no WCS is available but matched ``(ra, dec)`` and
    ``(x, y)`` samples are. For a single detector the affine fit is
    typically accurate to well under a pixel -- enough for visualization
    but not for science.

    Parameters
    ----------
    ra, dec : array-like
        Sky coordinates of the samples, in **radians**.
    x, y : array-like
        Detector pixel coordinates of the same samples.

    Returns
    -------
    sky_to_xy : callable
        Maps an `lsst.geom.SpherePoint` to an ``(x, y)`` tuple of
        `float` in the same pixel system as the input ``x, y``.
    """
    ra = np.asarray(ra)
    dec = np.asarray(dec)
    A = np.column_stack([np.ones(ra.size), ra, dec])
    coef_x, *_ = np.linalg.lstsq(A, np.asarray(x), rcond=None)
    coef_y, *_ = np.linalg.lstsq(A, np.asarray(y), rcond=None)

    def sky_to_xy(sphere_point):
        r = sphere_point.getRa().asRadians()
        d = sphere_point.getDec().asRadians()
        return (float(coef_x[0] + coef_x[1]*r + coef_x[2]*d),
                float(coef_y[0] + coef_y[1]*r + coef_y[2]*d))

    return sky_to_xy


def compute_tract_patch_outlines(skymap, sky_to_xy, sky_corners, clip_rect):
    """Project overlapping tract/patch boundaries into display coordinates.

    Parameters
    ----------
    skymap : `lsst.skymap.BaseSkyMap`
        Skymap to query for overlapping tracts and patches.
    sky_to_xy : callable
        Maps an `lsst.geom.SpherePoint` to an ``(x, y)`` tuple in the
        display's pixel coordinate system.
    sky_corners : `list` [`lsst.geom.SpherePoint`]
        Sky positions spanning the region of interest; passed to
        ``skymap.findTractPatchList`` to enumerate the tracts/patches that
        touch it.
    clip_rect : `tuple` [`float`]
        ``(xmin, xmax, ymin, ymax)`` display-pixel rectangle. Used only to
        rank tracts by their visible (clipped) patch area, so that the
        most-covering tract sorts first; it does not clip the returned
        geometry.

    Returns
    -------
    outlines : `list` [`dict`]
        One entry per overlapping tract, sorted by descending visible
        area (ties broken by tract id for determinism). Each is::

            {"tract_id": int,
             "rank": int,        # 0-based position in the sorted list
             "patches": [{"patch_index": int,           # sequential index
                          "corners_xy": [(x, y), ...],   # 4 corners
                          "center_xy": (x, y)},          # inner-bbox center
                         ...]}
    """
    import lsst.geom as geom

    xmin, xmax, ymin, ymax = clip_rect
    tract_patch_list = skymap.findTractPatchList(sky_corners)

    tract_entries = []  # (visible_area, tract_id, patches)
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
            xy_corners = [sky_to_xy(tract_wcs.pixelToSky(p)) for p in corners_px]
            center_px = geom.Point2D(0.5*(bbox.minX + bbox.maxX),
                                     0.5*(bbox.minY + bbox.maxY))
            center_xy = sky_to_xy(tract_wcs.pixelToSky(center_px))

            clipped = _clip_polygon_to_rect(xy_corners, xmin, xmax, ymin, ymax)
            per_tract_area += _polygon_area(clipped)
            per_tract_patches.append({"patch_index": patch.getSequentialIndex(),
                                      "corners_xy": xy_corners,
                                      "center_xy": center_xy})
        tract_entries.append((per_tract_area, tract_info.getId(), per_tract_patches))

    # Sort by descending visible area; break ties by tract id so the order
    # (and hence the mpl linestyle assignment) is stable across calls.
    tract_entries.sort(key=lambda e: (-e[0], e[1]))
    return [{"tract_id": tract_id, "rank": rank, "patches": patches}
            for rank, (_area, tract_id, patches) in enumerate(tract_entries)]


def draw_skymap_outlines_mpl(ax, skymap, sky_to_xy, sky_corners, *,
                             label_fontsize=7):
    """Overlay patch boundaries (with ``tract,patch`` labels) on a panel.

    Each overlapping patch gets a thin white outline (black-stroked so it
    reads on any colormap) plus a ``tract,patch`` label placed along the
    midpoint of its longest visible edge inside the current view.

    When more than one tract overlaps the panel (e.g. detectors near a
    tract boundary), patches are distinguished by linestyle: tracts are
    ranked by their total visible patch area and assigned solid, dashed,
    dotted, dash-dotted in turn.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Axis to draw on. Its current ``xlim``/``ylim`` define the visible
        region and are restored on return.
    skymap : `lsst.skymap.BaseSkyMap`
        Skymap to query for overlapping tracts and patches.
    sky_to_xy : callable
        Maps an `lsst.geom.SpherePoint` to an ``(x, y)`` tuple in the
        axis' data coordinate system (see `make_affine_sky_to_xy`).
    sky_corners : `list` [`lsst.geom.SpherePoint`]
        Sky positions spanning the region of interest, used to enumerate
        the overlapping tracts/patches.
    label_fontsize : `int` or `float`, optional
        Font size for the per-patch ``tract,patch`` labels.
    """
    import matplotlib.patheffects as pe

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xmin, xmax = xlim
    ymin, ymax = ylim
    clip_rect = (xmin, xmax, ymin, ymax)

    outlines = compute_tract_patch_outlines(skymap, sky_to_xy, sky_corners, clip_rect)

    # White lines with a thin black stroke read on any colormap background.
    line_outline = [pe.withStroke(linewidth=2.0, foreground="black")]
    text_outline = [pe.withStroke(linewidth=1.4, foreground="black")]
    linestyles = ("-", "--", ":", "-.")

    for tract in outlines:
        rank = tract["rank"]
        tract_id = tract["tract_id"]
        linestyle = linestyles[rank] if rank < len(linestyles) else "-"
        patch_kwargs = dict(color="white", linewidth=0.5, alpha=0.85,
                            linestyle=linestyle,
                            path_effects=line_outline, zorder=6)
        for patch in tract["patches"]:
            xy_corners = patch["corners_xy"]
            xs_c = [p[0] for p in xy_corners] + [xy_corners[0][0]]
            ys_c = [p[1] for p in xy_corners] + [xy_corners[0][1]]
            ax.plot(xs_c, ys_c, **patch_kwargs)

            # Place the label at the midpoint of the patch edge with the
            # longest visible portion within the panel, offset slightly
            # toward the patch center so the text sits inside.
            cx, cy = patch["center_xy"]
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

            ax.text(mx, my, f"{tract_id},{patch['patch_index']}",
                    ha="center", va="center",
                    rotation=angle_deg, rotation_mode="anchor",
                    color="white", fontsize=label_fontsize,
                    path_effects=text_outline, zorder=7,
                    clip_on=True)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def draw_skymap_outlines_afw(afw_display, skymap, wcs, bbox, *,
                             ctype="green", label_size=1.5, draw_labels=True):
    """Overlay tract/patch boundaries on the current `afw.display` frame.

    Each overlapping patch is drawn as a closed polyline in the image's
    parent pixel coordinates and, optionally, labeled ``tract,patch``
    near the center of its visible portion.

    Parameters
    ----------
    afw_display : `lsst.afw.display.Display`
        Display whose current frame already shows the exposure. The caller
        is responsible for selecting the frame.
    skymap : `lsst.skymap.BaseSkyMap`
        Skymap to query for overlapping tracts and patches.
    wcs : `lsst.afw.geom.SkyWcs`
        WCS of the displayed exposure (``exposure.wcs``).
    bbox : `lsst.geom.Box2I` or `lsst.geom.Box2D`
        Parent bounding box of the displayed exposure.
        Defines the footprint searched for overlapping patches
        and the region used to place labels.
    ctype : `str`, optional
        Display color for both the outlines and labels.
    label_size : `float`, optional
        Text size for the ``tract,patch`` labels.
    draw_labels : `bool`, optional
        If False, draw only the outlines.
    """
    import lsst.geom as geom

    def sky_to_xy(sphere_point):
        p = wcs.skyToPixel(sphere_point)
        return (p.getX(), p.getY())

    xmin = bbox.getMinX()
    xmax = bbox.getMaxX()
    ymin = bbox.getMinY()
    ymax = bbox.getMaxY()
    clip_rect = (xmin, xmax, ymin, ymax)

    # Sky positions at the four image corners span the footprint for the
    # tract/patch lookup.
    corner_px = [geom.Point2D(xmin, ymin), geom.Point2D(xmax, ymin),
                 geom.Point2D(xmax, ymax), geom.Point2D(xmin, ymax)]
    sky_corners = [wcs.pixelToSky(p) for p in corner_px]

    outlines = compute_tract_patch_outlines(skymap, sky_to_xy, sky_corners, clip_rect)

    with afw_display.Buffering():
        for tract in outlines:
            tract_id = tract["tract_id"]
            for patch in tract["patches"]:
                xy_corners = patch["corners_xy"]
                # Closed polyline: repeat the first corner.
                afw_display.line(list(xy_corners) + [xy_corners[0]], ctype=ctype)
                if not draw_labels:
                    continue
                # Anchor the label at the centroid of the patch's visible
                # portion so it doesn't land off-image for patches that
                # mostly fall outside the detector.
                clipped = _clip_polygon_to_rect(xy_corners, xmin, xmax, ymin, ymax)
                if clipped:
                    lx = sum(p[0] for p in clipped)/len(clipped)
                    ly = sum(p[1] for p in clipped)/len(clipped)
                else:
                    lx, ly = patch["center_xy"]
                afw_display.dot(f"{tract_id},{patch['patch_index']}",
                                lx, ly, size=label_size, ctype=ctype)


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

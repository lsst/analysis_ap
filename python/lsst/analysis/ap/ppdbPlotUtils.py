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

"""Plotting helpers for exploring the contents of the PPDB.

These build on the `lsst.analysis.ap.ppdb.PpdbTap` loader to visualize what
is currently in the prototype PPDB. They are intended for interactive use
(e.g. in a notebook), so they draw into the active Matplotlib figure and
return the `skyproj` projection for further customization.
"""

__all__ = ["plot_ppdb_sky_density"]

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import skyproj

import astropy.units as u
from astropy.coordinates import SkyCoord

from .ppdb import PpdbTap, DEFAULT_PPDB_TAP_URL


def _wrap_ra(ra):
    """Wrap right ascension (deg) to [-180, 180), placing RA=0 at center."""
    return (np.asarray(ra, dtype=float) + 180.0) % 360.0 - 180.0


def _filter_column(table, min_n_dia_sources, min_reliability):
    """Resolve the per-table filter column and its threshold.

    Each table has one extra column worth fetching and cutting on: DiaObject
    on ``nDiaSources``, DiaSource on ``reliability``. This column is always
    fetched (so it comes back in the returned catalog and the cut can be
    re-applied client-side), while the threshold is applied only for display.
    Each ``min_*`` filter belongs to exactly one table; asking for the other
    table's filter raises rather than being silently ignored.

    Returns
    -------
    column : `str`
        Name of the column to fetch and threshold on.
    threshold : `float` or `None`
        The minimum value to keep, or None for no cut.
    """
    if table == "DiaObject":
        if min_reliability is not None:
            raise ValueError("min_reliability applies to DiaSource, not "
                             "DiaObject.")
        return "nDiaSources", min_n_dia_sources
    if table == "DiaSource":
        if min_n_dia_sources is not None:
            raise ValueError("min_n_dia_sources applies to DiaObject, not "
                             "DiaSource.")
        return "reliability", min_reliability
    raise ValueError(f"Unknown table {table!r}; expected 'DiaObject' or "
                     "'DiaSource'.")


def _density_hpxmap(ra, dec, nside):
    """Bin points into a HEALPix source-density map (# / deg^2).

    HEALPix pixels have equal solid angle, so dividing the per-pixel counts by
    the pixel area gives a projection-independent sky density; the cos(dec)
    area factor is handled by the equal-area binning rather than by hand.

    Returns
    -------
    density : `numpy.ndarray`
        Full-sky ring-ordered map; empty pixels are set to ``healpy.UNSEEN``.
    """
    pix = hp.ang2pix(nside, np.radians(90.0 - dec), np.radians(ra % 360.0))
    counts = np.bincount(pix, minlength=hp.nside2npix(nside)).astype(float)
    density = counts / hp.nside2pixarea(nside, degrees=True)
    density[counts == 0] = hp.UNSEEN
    return density


def _draw_matched_colorbar(sp, mappable, label, pad=0.015, width=0.018):
    """Add a colorbar whose vertical extent matches the drawn map.

    The projection axes has a fixed aspect, so the map fills only part of the
    box allocated to the axes. A default colorbar is sized from that whole box
    and would tower over the map, so the colorbar axes is instead placed at the
    aspect-adjusted position (figure coordinates), matching the map's height.

    Parameters
    ----------
    sp : `skyproj.Skyproj`
        Projection the colorbar describes.
    mappable : `matplotlib.cm.ScalarMappable`
        The mapped artist (the density mesh) to draw a colorbar for.
    label : `str`
        Colorbar label.
    pad, width : `float`, optional
        Gap between map and colorbar, and colorbar width, as fractions of the
        figure width.

    Returns
    -------
    cax : `matplotlib.axes.Axes`
        The colorbar axes.
    """
    fig = sp.ax.figure
    sp.ax.apply_aspect()   # make sure the active position is up to date
    pos = sp.ax.get_position()
    cax = fig.add_axes([pos.x1 + pad, pos.y0, width, pos.height])
    fig.colorbar(mappable, cax=cax, label=label)
    return cax


def _draw_lat_circle(sp, frame, lat, **plot_kwargs):
    """Draw a constant-latitude circle of ``frame`` on the sky projection.

    The circle is sampled in longitude, converted to ICRS, and drawn through
    the projection (which handles the RA wrap), so it appears as the correct
    curve for the map.

    Parameters
    ----------
    sp : `skyproj.Skyproj`
        Projection to draw into.
    frame : `str`
        ``"galactic"`` or ``"ecliptic"``.
    lat : `float`
        Latitude of the circle in that frame, in degrees (0 = the plane).
    **plot_kwargs
        Forwarded to the projection's ``plot`` (color, lw, ls, label, ...).
    """
    lon = np.linspace(0.0, 360.0, 1441)
    lats = np.full_like(lon, lat)
    if frame == "galactic":
        coord = SkyCoord(l=lon * u.deg, b=lats * u.deg, frame="galactic")
    elif frame == "ecliptic":
        coord = SkyCoord(lon=lon * u.deg, lat=lats * u.deg,
                         frame="barycentrictrueecliptic")
    else:
        raise ValueError(f"Unknown frame {frame!r}; expected 'galactic' or "
                         "'ecliptic'.")
    icrs = coord.icrs
    sp.ax.plot(icrs.ra.deg, icrs.dec.deg, **plot_kwargs)


# Named LSST fields to mark, each a list of (RA, Dec) pointings in ICRS
# degrees. The first five are the Deep Drilling Fields (EDFS being a single
# field observed at two pointings, "a" and "b"); M49 is a commissioning
# target field, at the position of the galaxy itself.
_LABELED_FIELDS = {
    "ELAISS1": [(9.45, -44.02)],
    "XMM_LSS": [(35.57, -4.82)],
    "ECDFS": [(52.98, -28.12)],
    "COSMOS": [(150.11, 2.23)],
    "EDFS": [(58.9, -49.32), (63.6, -47.6)],
    "M49": [(187.44, 8.00)],
}

# Radius (deg) of a circle whose area equals the LSST field of view
# (~9.6 deg^2). In an equal-area projection a fixed-angular-radius circle has
# the FoV solid angle at every declination, so no cos(dec) correction is
# needed.
_LSST_FOV_RADIUS_DEG = (9.6 / np.pi) ** 0.5


def _label_fields(sp, ra_range, dec_range):
    """Mark the named LSST fields within the window at LSST-FoV scale.

    Each field's pointings are drawn as circles of the LSST field-of-view
    angular radius (EDFS is the union of its two pointings), with one name
    label per field. Labels sit just above the topmost circle of their field,
    offset in declination so they clear the field rather than covering it.
    Pointings outside the window are skipped.
    """
    (ra_lo, ra_hi), (dec_lo, dec_hi) = ra_range, dec_range
    for name, pointings in _LABELED_FIELDS.items():
        shown = [(ra, dec) for ra, dec in pointings
                 if ra_lo <= float(_wrap_ra(ra)) <= ra_hi
                 and dec_lo <= dec <= dec_hi]
        if not shown:
            continue
        for ra, dec in shown:
            sp.ax.circle(ra, dec, _LSST_FOV_RADIUS_DEG, color="black", lw=1.2,
                         zorder=6)
        # One label per field, centered in RA and raised clear of the topmost
        # circle. The offset is in degrees, so it tracks the circles as the
        # map is zoomed.
        lra = sum(ra for ra, _ in shown) / len(shown)
        ldec = max(dec for _, dec in shown) + _LSST_FOV_RADIUS_DEG + 0.5
        sp.ax.text(lra, ldec, name, transform=sp.crs, fontsize=8,
                   color="black", ha="center", va="bottom", zorder=7,
                   bbox=dict(boxstyle="round,pad=0.15", fc="white",
                             ec="none", alpha=0.7))


def plot_ppdb_sky_density(table="DiaObject", *, ppdb=None, catalog=None,
                          url=DEFAULT_PPDB_TAP_URL, nside=256,
                          maxrec=50_000_000, min_n_dia_sources=None,
                          min_reliability=None, band_lines=(10, -10),
                          draw_ecliptic=False, ecliptic_band_lines=(),
                          label_fields=True, ra_lim=None, dec_lim=(-60, 30),
                          projection=None, lon_0=0.0, figsize=(12, 6),
                          dpi=150, ax=None):
    """Sky-density map of a PPDB table on an equal-area projection.

    Bins a PPDB table into HEALPix pixels and draws the source density
    (# / deg^2, log color scale) on a `skyproj` equal-area map, with the
    galactic plane, an optional ecliptic, and the LSST Deep Drilling Fields
    overlaid. Because the projection is equal-area, the cos(dec) area factor
    is incorporated by the map itself.

    Parameters
    ----------
    table : `str`
        PPDB table to plot, ``"DiaObject"`` or ``"DiaSource"``. DiaObjects
        are filtered to their current version (``validityEndMjdTai IS NULL``).
    ppdb : `lsst.analysis.ap.ppdb.PpdbTap`, optional
        Reuse an existing loader (avoids re-authenticating); built from
        ``url`` if not given. Needs ``RSP_TOKEN`` in the environment.
    catalog : `astropy.table.Table`, optional
        A previously returned catalog (columns ``ra``, ``dec``, and the
        table's filter column). If given, no query is run -- fetch the full
        table once, then re-plot (zoom, change thresholds) for free. Its rows
        are the full, unfiltered result; ``min_*`` cuts are applied per call.
    url : `str`, optional
        PPDB TAP endpoint (defaults to the prototype at data-int).
    nside : `int`, optional
        HEALPix resolution of the density map. Higher is finer (and noisier
        where sparse); 256 gives ~13.7 arcmin pixels.
    maxrec : `int`, optional
        Server-side row cap; set above the table size to avoid truncation.
    min_n_dia_sources : `int`, optional
        DiaObject only: show only objects with
        ``nDiaSources >= min_n_dia_sources``. Applied client-side, so it can
        be changed freely when re-plotting from a cached ``catalog``.
    min_reliability : `float`, optional
        DiaSource only: show only sources with
        ``reliability >= min_reliability``. Client-side, like
        ``min_n_dia_sources``. NaN reliabilities are dropped.
    band_lines : iterable of `float`, optional
        Galactic latitudes (deg) to draw as dashed lines; defaults to
        ``(10, -10)`` to bracket the |b|<10 plane band. Pass ``()`` for none.
    draw_ecliptic : `bool`, optional
        If True, also draw the ecliptic plane (ecliptic latitude 0) as a solid
        line in a color distinct from the galactic plane. Off by default.
    ecliptic_band_lines : iterable of `float`, optional
        Ecliptic latitudes (deg) to draw as dashed lines, the ecliptic analog
        of ``band_lines`` (e.g. ``(10, -10)``). Only drawn when
        ``draw_ecliptic`` is True. Empty by default.
    label_fields : `bool`, optional
        If True (default), mark and name the LSST fields that fall within the
        plot window (the Deep Drilling Fields plus M49).
    ra_lim, dec_lim : `tuple` [`float`, `float`], optional
        ``(min, max)`` map extent in degrees. RA is centered on ``lon_0``;
        ``ra_lim`` is in the [-180, 180) convention (RA=0 at center) and
        defaults to the full ``(-180, 180)``. ``dec_lim`` defaults to
        ``(-60, 30)``; pass ``(-90, 90)`` for full Dec.
    projection : `type`, optional
        A `skyproj.Skyproj` subclass to use; defaults to
        `skyproj.McBrydeSkyproj` (McBryde-Thomas Flat Polar Quartic, an
        equal-area world projection). Any equal-area skyproj projection keeps
        the density interpretation valid.
    lon_0 : `float`, optional
        Central longitude (RA, deg) of the projection; defaults to 0.
    figsize : `tuple` [`float`, `float`], optional
        Size of a newly created figure, in inches. Ignored when ``ax`` given.
    dpi : `float`, optional
        Resolution of a newly created figure. Ignored when ``ax`` is given.
    ax : `matplotlib.axes.Axes`, optional
        Axes to build the projection on; a new figure is made if not given.

    Returns
    -------
    sp : `skyproj.Skyproj`
        The projection drawn into (``sp.ax`` is the underlying axes). Pass its
        axes back as ``ax=`` only if you know it is already a skyproj axes.
    catalog : `astropy.table.Table`
        The full fetched table (``ra``, ``dec``, and the filter column). Pass
        it back as ``catalog=`` to re-plot (zoom, retune ``min_*``) without
        re-querying.
    """
    column, threshold = _filter_column(table, min_n_dia_sources,
                                       min_reliability)

    # Fetch ra/dec plus the filter column unless a cached catalog was given.
    # The whole (unfiltered) table is returned so thresholds can be retuned
    # client-side; only the current version of each DiaObject is fetched.
    if catalog is None:
        if ppdb is None:
            ppdb = PpdbTap(url=url)
        where = " WHERE validityEndMjdTai IS NULL" if table == "DiaObject" else ""
        catalog = ppdb.run_query(
            f"SELECT ra, dec, {column} FROM ppdb.{table}{where}", maxrec=maxrec)

    ra = np.asarray(catalog["ra"], dtype=float)
    dec = np.asarray(catalog["dec"], dtype=float)

    # Select the rows to bin: finite coords, the threshold cut, and the map
    # window (RA compared in the centered [-180, 180) convention), so the
    # density map and its color scale reflect only what is shown.
    ra_lo, ra_hi = ra_lim if ra_lim is not None else (-180.0, 180.0)
    dec_lo, dec_hi = dec_lim if dec_lim is not None else (-90.0, 90.0)
    ra_wrapped = _wrap_ra(ra)
    sel = (np.isfinite(ra) & np.isfinite(dec)
           & (ra_wrapped >= ra_lo) & (ra_wrapped <= ra_hi)
           & (dec >= dec_lo) & (dec <= dec_hi))
    if threshold is not None:
        sel &= np.asarray(catalog[column], dtype=float) >= threshold
    n_shown = int(sel.sum())

    if ax is None:
        _, ax = plt.subplots(figsize=figsize, dpi=dpi)

    projection = projection if projection is not None else skyproj.McBrydeSkyproj
    sp = projection(ax=ax, lon_0=lon_0)

    # Equal-area HEALPix density map, log-scaled. The colorbar is added at the
    # end, once the final extent fixes the map's on-figure height.
    density_mesh = None
    if n_shown:
        density = _density_hpxmap(ra[sel], dec[sel], nside)
        density_mesh = sp.draw_hpxmap(density, norm="log", cmap="viridis")[0]

    # Galactic plane (b=0), its optional latitude band, and the galactic
    # center marker.
    _draw_lat_circle(sp, "galactic", 0.0, color="crimson", lw=1.5,
                     label="Galactic plane")
    for b in band_lines:
        _draw_lat_circle(sp, "galactic", b, color="crimson", lw=0.8, ls="--",
                         alpha=0.6)
    gc = SkyCoord(l=0 * u.deg, b=0 * u.deg, frame="galactic").icrs
    sp.ax.scatter(gc.ra.deg, gc.dec.deg, marker="*", s=320, color="crimson",
                  edgecolors="white", linewidths=0.8, zorder=5,
                  label="Galactic center")

    # Ecliptic plane and its optional latitude band, in a distinct color.
    if draw_ecliptic:
        _draw_lat_circle(sp, "ecliptic", 0.0, color="dodgerblue", lw=1.5,
                         label="Ecliptic")
        for b in ecliptic_band_lines:
            _draw_lat_circle(sp, "ecliptic", b, color="dodgerblue", lw=0.8,
                             ls="--", alpha=0.6)

    # Named-field FoV circles and labels.
    if label_fields:
        _label_fields(sp, (ra_lo, ra_hi), (dec_lo, dec_hi))

    sp.set_extent([ra_hi, ra_lo, dec_lo, dec_hi])  # RA increases to the left
    sp.ax.legend(loc="lower right", framealpha=0.9)
    if density_mesh is not None:
        _draw_matched_colorbar(sp, density_mesh,
                               "Source density (#/degree$^2$)")
    # The colorbar is on the right, so the top is free for a title; pad it
    # clear of the RA tick labels skyproj draws along the top edge.
    sp.ax.set_title(f"PPDB {table}  (N={n_shown:,})", pad=24)
    return sp, catalog

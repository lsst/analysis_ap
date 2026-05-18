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

"""Catalog cross-matching and pair-wise comparison utilities for
DiaSource-like tables.
"""

__all__ = ["match_catalogs", "flux_residuals", "match_to_truth"]

import numpy as np
import pandas as pd

import astropy.units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky


def _as_arcsec(radius):
    """Coerce a number-or-Quantity radius to a float in arcseconds."""
    if isinstance(radius, u.Quantity):
        return float(radius.to_value(u.arcsec))
    return float(radius)


def _group_keys(srcs1, srcs2, on):
    """Yield the union of distinct ``on``-tuples present in either frame."""
    if not on:
        yield None
        return
    s1 = set(map(tuple, srcs1[list(on)].itertuples(index=False, name=None)))
    s2 = set(map(tuple, srcs2[list(on)].itertuples(index=False, name=None)))
    for key in sorted(s1 | s2):
        yield key


def _select_group(df, on, key):
    """Return the slice of ``df`` whose ``on`` columns equal ``key``."""
    if key is None or not on:
        return df
    mask = pd.Series(True, index=df.index)
    for col, val in zip(on, key):
        mask &= df[col] == val
    return df[mask]


def match_catalogs(srcs1, srcs2, radius=0.5*u.arcsec, on=("visit", "detector"),
                   ra_col="ra", dec_col="dec", id_col="diaSourceId"):
    """Spatially cross-match two DiaSource-like DataFrames.

    Sources are first partitioned by the columns in
    ``on`` (e.g. matched only within the same (visit, detector)), then matched
    via nearest-neighbor on the sphere.

    Parameters
    ----------
    srcs1, srcs2 : `pandas.DataFrame`
        Source tables. Each must contain ``ra_col``, ``dec_col``, ``id_col``,
        and every column listed in ``on``.
    radius : `astropy.units.Quantity` or `float`
        Maximum separation for a pair to count as matched. A bare float is
        interpreted as arcseconds.
    on : `tuple` [`str`]
        Columns to group on before matching. Pass an empty tuple to match the
        full catalog with no grouping.
    ra_col, dec_col : `str`
        Column names for sky coordinates, in degrees.
    id_col : `str`
        Column name for the source id in both catalogs.

    Returns
    -------
    matched : `pandas.DataFrame`
        Rows from ``srcs1`` that found a partner in ``srcs2`` within
        ``radius``. Two columns are added: ``<id_col>_2`` with the partner's
        id, and ``xmatch_dist_arcsec`` with the on-sky separation in arcsec.
    unique1 : `pandas.DataFrame`
        Rows from ``srcs1`` with no partner.
    unique2 : `pandas.DataFrame`
        Rows from ``srcs2`` not pointed at by any matched pair.
    """
    rad_arcsec = _as_arcsec(radius)
    on = tuple(on)
    id2_col = f"{id_col}_2"

    matched_chunks = []
    unique1_chunks = []
    unique2_chunks = []

    for key in _group_keys(srcs1, srcs2, on):
        gs1 = _select_group(srcs1, on, key).copy()
        gs2 = _select_group(srcs2, on, key).copy()

        if len(gs1) == 0:
            unique2_chunks.append(gs2)
            continue
        if len(gs2) == 0:
            unique1_chunks.append(gs1)
            continue

        coords1 = SkyCoord(ra=gs1[ra_col].values*u.deg,
                           dec=gs1[dec_col].values*u.deg)
        coords2 = SkyCoord(ra=gs2[ra_col].values*u.deg,
                           dec=gs2[dec_col].values*u.deg)
        idx, sep, _ = match_coordinates_sky(coords1, coords2)

        gs1["xmatch_dist_arcsec"] = sep.to_value(u.arcsec)
        gs1[id2_col] = gs2[id_col].values[idx]

        has_match = gs1["xmatch_dist_arcsec"] <= rad_arcsec
        m = gs1[has_match]
        u1 = gs1[~has_match].drop(columns=["xmatch_dist_arcsec", id2_col])
        u2 = gs2[~gs2[id_col].isin(set(m[id2_col]))]

        matched_chunks.append(m)
        unique1_chunks.append(u1)
        unique2_chunks.append(u2)

    matched = (pd.concat(matched_chunks) if matched_chunks
               else srcs1.iloc[0:0].assign(**{"xmatch_dist_arcsec": np.nan,
                                              id2_col: pd.NA}))
    unique1 = pd.concat(unique1_chunks) if unique1_chunks else srcs1.iloc[0:0].copy()
    unique2 = pd.concat(unique2_chunks) if unique2_chunks else srcs2.iloc[0:0].copy()
    return matched, unique1, unique2


def flux_residuals(matched, srcs2, flux_col="psfFlux", err_col="psfFluxErr",
                   id_col="diaSourceId", plot=False):
    """Compute per-pair flux residuals from a `match_catalogs` result.

    Parameters
    ----------
    matched : `pandas.DataFrame`
        Output of `match_catalogs` (rows from catalog 1 with partner ids).
    srcs2 : `pandas.DataFrame`
        Catalog 2, indexed implicitly by ``id_col`` for partner lookup.
    flux_col, err_col : `str`
        Column names for the flux and its error in both catalogs.
    id_col : `str`
        Source-id column in both catalogs. ``matched`` is assumed to have
        ``f"{id_col}_2"`` populated by `match_catalogs`.
    plot : `bool`
        If True, return a histogram + Q-Q plot in addition to the residuals.

    Returns
    -------
    residuals : `pandas.DataFrame`
        One row per pair with columns ``flux1``, ``flux2``, ``err1``, ``err2``,
        ``delta_flux``, ``delta_flux_sigma``, and ``xmatch_dist_arcsec``.
    fig : `matplotlib.figure.Figure`, optional
        Only returned when ``plot=True``.
    """
    id2_col = f"{id_col}_2"
    s2 = srcs2.set_index(id_col)
    partner_ids = matched[id2_col].values

    f1 = matched[flux_col].to_numpy(dtype=float, copy=True)
    e1 = matched[err_col].to_numpy(dtype=float, copy=True)
    f2 = s2.loc[partner_ids, flux_col].to_numpy(dtype=float, copy=True)
    e2 = s2.loc[partner_ids, err_col].to_numpy(dtype=float, copy=True)

    delta = f1 - f2
    sigma = np.sqrt(e1**2 + e2**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta_sigma = np.where(sigma > 0, delta / sigma, np.nan)

    residuals = pd.DataFrame({
        id_col: matched[id_col].values,
        id2_col: partner_ids,
        "flux1": f1,
        "flux2": f2,
        "err1": e1,
        "err2": e2,
        "delta_flux": delta,
        "delta_flux_sigma": delta_sigma,
        "xmatch_dist_arcsec": matched["xmatch_dist_arcsec"].values,
    })

    if not plot:
        return residuals
    return residuals, _plot_flux_residuals(residuals, flux_col)


def _plot_flux_residuals(residuals, flux_col):
    """Histogram + Q-Q plot of normalized flux residuals."""
    import matplotlib.pyplot as plt
    sigma = residuals["delta_flux_sigma"].dropna().values
    if len(sigma) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no finite residuals", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(sigma, bins=50, color="C0")
    axes[0].axvline(0, color="grey", lw=0.5)
    axes[0].set_xlabel(rf"$\Delta {flux_col} / \sigma$")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"N={len(sigma)}, "
                      f"med={np.median(sigma):.3f}, "
                      f"MAD={np.median(np.abs(sigma - np.median(sigma))):.3f}")

    # Quick Q-Q vs standard normal without depending on scipy.
    sp = np.sort(sigma)
    # Inverse-CDF approximation for the standard normal via erfinv.
    quantiles = (np.arange(len(sp)) + 0.5) / len(sp)
    expected = np.sqrt(2) * _erfinv(2*quantiles - 1)
    axes[1].plot(expected, sp, ".", ms=2)
    lim = max(abs(expected[0]), abs(expected[-1]), 3.0)
    axes[1].plot([-lim, lim], [-lim, lim], "k--", lw=0.5)
    axes[1].set_xlabel("expected (N(0,1))")
    axes[1].set_ylabel("observed")
    axes[1].set_title("Q-Q vs standard normal")
    fig.tight_layout()
    return fig


def _erfinv(y):
    """Approximate inverse error function (vectorized).

    Uses the formula from Winitzki (2008); accurate to ~4e-3 across the
    domain, which is plenty for plotting Q-Q lines.
    """
    a = 0.147
    sign = np.sign(y)
    ln1 = np.log(np.clip(1 - y*y, 1e-300, 1.0))
    term = 2/(np.pi*a) + ln1/2
    return sign * np.sqrt(np.sqrt(term*term - ln1/a) - term)


def match_to_truth(srcs, truth, radius=0.5*u.arcsec,
                   src_ra="ra", src_dec="dec", src_id="diaSourceId",
                   truth_ra="ra", truth_dec="dec", truth_id="injection_id"):
    """Match a detected catalog against a truth/injection catalog.

    Performs the match in both directions to compute purity (fraction of
    detections that correspond to a real injected source) and completeness
    (fraction of injected sources recovered).

    Parameters
    ----------
    srcs : `pandas.DataFrame`
        Detected sources.
    truth : `pandas.DataFrame`
        Truth catalog, e.g. an injection catalog.
    radius : `astropy.units.Quantity` or `float`
        Maximum separation for a match.
    src_ra, src_dec, src_id : `str`
        Column names in ``srcs``.
    truth_ra, truth_dec, truth_id : `str`
        Column names in ``truth``.

    Returns
    -------
    out : `dict`
        With keys:

        - ``"srcs"``: a copy of ``srcs`` with three columns appended:
          ``"is_real"`` (bool), ``"truth_dist_arcsec"`` (float), and
          ``f"{truth_id}_match"`` (matched truth id, NA if no match).
        - ``"truth"``: a copy of ``truth`` with three columns appended:
          ``"detected"``, ``"detection_dist_arcsec"``, and
          ``f"{src_id}_match"``.
        - ``"purity"``: float, fraction of ``srcs`` rows with ``is_real``.
        - ``"completeness"``: float, fraction of ``truth`` rows with
          ``detected``.
    """
    rad_arcsec = _as_arcsec(radius)
    srcs_out = srcs.copy()
    truth_out = truth.copy()

    truth_id_match = f"{truth_id}_match"
    src_id_match = f"{src_id}_match"

    if len(srcs_out) == 0 or len(truth_out) == 0:
        srcs_out["is_real"] = False
        srcs_out["truth_dist_arcsec"] = np.nan
        srcs_out[truth_id_match] = pd.NA
        truth_out["detected"] = False
        truth_out["detection_dist_arcsec"] = np.nan
        truth_out[src_id_match] = pd.NA
        return {"srcs": srcs_out, "truth": truth_out,
                "purity": 0.0, "completeness": 0.0}

    sc_src = SkyCoord(srcs_out[src_ra].values*u.deg,
                      srcs_out[src_dec].values*u.deg)
    sc_tru = SkyCoord(truth_out[truth_ra].values*u.deg,
                      truth_out[truth_dec].values*u.deg)

    idx_to_truth, sep_to_truth, _ = match_coordinates_sky(sc_src, sc_tru)
    srcs_out["truth_dist_arcsec"] = sep_to_truth.to_value(u.arcsec)
    srcs_out[truth_id_match] = truth_out[truth_id].values[idx_to_truth]
    srcs_out["is_real"] = srcs_out["truth_dist_arcsec"] <= rad_arcsec
    srcs_out.loc[~srcs_out["is_real"], truth_id_match] = pd.NA

    idx_to_src, sep_to_src, _ = match_coordinates_sky(sc_tru, sc_src)
    truth_out["detection_dist_arcsec"] = sep_to_src.to_value(u.arcsec)
    truth_out[src_id_match] = srcs_out[src_id].values[idx_to_src]
    truth_out["detected"] = truth_out["detection_dist_arcsec"] <= rad_arcsec
    truth_out.loc[~truth_out["detected"], src_id_match] = pd.NA

    purity = float(srcs_out["is_real"].sum() / len(srcs_out))
    completeness = float(truth_out["detected"].sum() / len(truth_out))
    return {"srcs": srcs_out, "truth": truth_out,
            "purity": purity, "completeness": completeness}

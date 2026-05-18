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

"""Pixel-level QA for AP difference images.

Two utilities live here:

- `image_diff_stats` reports basic statistics (median, MAD, stdev, kurtosis,
  mask-plane fractions) of a single difference image.
- `pixel_compare` aligns two difference images for the same data id and
  returns their pixel-wise difference, ratio, mask XOR, and a small summary.

Both are intended for human-driven analysis from a notebook. The first is
also designed to be mapped over a list of dataIds to produce a per-detector
quality DataFrame.
"""

from __future__ import annotations

__all__ = ["image_diff_stats", "image_diff_stats_table", "pixel_compare"]

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


def _finite_view(array):
    """Return only the finite entries of ``array`` as a flat 1-D view."""
    flat = np.asarray(array).ravel()
    return flat[np.isfinite(flat)]


def _basic_stats(array):
    """Median, MAD, stdev, and kurtosis of the finite entries of ``array``.

    Kurtosis is the Pearson definition (``E[((x-mu)/sigma)**4] - 3``); zero
    for a Gaussian. Computed without scipy so this module has no extra deps.
    """
    finite = _finite_view(array)
    if finite.size == 0:
        return {"median": np.nan, "mad": np.nan, "stddev": np.nan, "kurtosis": np.nan}
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    stddev = float(np.std(finite))
    if stddev > 0:
        z = (finite - finite.mean()) / stddev
        kurtosis = float(np.mean(z**4) - 3.0)
    else:
        kurtosis = np.nan
    return {"median": median, "mad": mad, "stddev": stddev, "kurtosis": kurtosis}


def _mask_plane_fractions(mask):
    """Return ``{plane_name: fraction_of_pixels_set}`` for an afw mask.

    Parameters
    ----------
    mask : `lsst.afw.image.Mask`
        The mask whose planes are to be summed.

    Returns
    -------
    fractions : `dict` [`str`, `float`]
        Per-plane fraction of pixels where that plane bit is set.
    """
    arr = mask.array
    npix = arr.size
    if npix == 0:
        return {}
    out = {}
    for plane_name, plane_bit in mask.getMaskPlaneDict().items():
        bitmask = np.uint64(1) << np.uint64(plane_bit)
        out[f"frac_{plane_name}"] = float(np.sum((arr & bitmask) != 0) / npix)
    return out


def image_diff_stats(butler, visit, detector, dataset_name="difference_image"):
    """Summary statistics on a single difference image.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler initialized with the relevant collections.
    visit, detector : `int`
        Data id selecting the exposure to load.
    dataset_name : `str`, optional
        Butler dataset type name to load. Defaults to ``"difference_image"``.

    Returns
    -------
    stats : `dict`
        ``{"visit", "detector", "median", "mad", "stddev", "kurtosis"}`` plus
        a ``frac_<PLANE>`` entry per mask plane on the exposure.
    """
    diff = butler.get(dataset_name, {"visit": visit, "detector": detector})
    stats = {"visit": visit, "detector": detector}
    stats.update(_basic_stats(diff.image.array))
    stats.update(_mask_plane_fractions(diff.mask))
    return stats


def image_diff_stats_table(butler,
                           data_ids: Iterable[Mapping[str, int]],
                           dataset_name="difference_image"):
    """Run `image_diff_stats` over many data ids and return a DataFrame.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
    data_ids : iterable of mapping
        Each mapping must contain at least ``visit`` and ``detector`` keys.
    dataset_name : `str`, optional
        See `image_diff_stats`.

    Returns
    -------
    table : `pandas.DataFrame`
        One row per dataId, indexed by ``(visit, detector)``. Mask-plane
        columns may be missing on some rows if the corresponding plane is not
        defined on every exposure; pandas fills those with NaN.
    """
    rows = []
    for data_id in data_ids:
        rows.append(image_diff_stats(butler,
                                     data_id["visit"],
                                     data_id["detector"],
                                     dataset_name=dataset_name))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index(["visit", "detector"]).sort_index()


@dataclass
class PixelCompareResult:
    """Container for `pixel_compare` outputs.

    Attributes
    ----------
    img1, img2 : `lsst.afw.image.Exposure`
        The two loaded exposures, in case the caller wants their masks/WCS.
    diff : `numpy.ndarray`
        Pixel array of ``img1 - img2``.
    ratio : `numpy.ndarray`
        Pixel array of ``img1 / img2``; NaN where ``img2 == 0``.
    mask_diff : `numpy.ndarray`
        XOR of the two mask arrays: nonzero pixels are where the mask planes
        differ between the two images.
    summary : `dict`
        ``{"visit", "detector", "diff_median", "diff_mad", "diff_stddev",
        "n_mask_pixels_changed", "frac_mask_pixels_changed"}``.
    """
    img1: object
    img2: object
    diff: np.ndarray
    ratio: np.ndarray
    mask_diff: np.ndarray
    summary: dict


def pixel_compare(butler1, butler2, visit, detector,
                  dataset_name="difference_image"):
    """Compare two difference images for the same data id, pixel by pixel.

    Pull the same dataId from two butlers (or two collections of one butler)
    and inspect the residuals.

    Parameters
    ----------
    butler1, butler2 : `lsst.daf.butler.Butler`
        Two butlers, possibly the same instance with different collections.
    visit, detector : `int`
        Data id to load from both butlers.
    dataset_name : `str`, optional
        Butler dataset type name to load from each butler. Defaults to
        ``"difference_image"``.

    Returns
    -------
    result : `PixelCompareResult`

    Raises
    ------
    ValueError
        If the two loaded images differ in shape (cannot be aligned by simple
        subtraction).
    """
    img1 = butler1.get(dataset_name, {"visit": visit, "detector": detector})
    img2 = butler2.get(dataset_name, {"visit": visit, "detector": detector})

    a1 = img1.image.array
    a2 = img2.image.array
    if a1.shape != a2.shape:
        raise ValueError(f"Image shapes differ for visit={visit} detector={detector}: "
                         f"{a1.shape} vs {a2.shape}")

    diff = a1 - a2
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(a2 != 0, a1 / a2, np.nan)

    mask_diff = img1.mask.array ^ img2.mask.array

    finite = _finite_view(diff)
    if finite.size:
        diff_median = float(np.median(finite))
        diff_mad = float(np.median(np.abs(finite - diff_median)))
        diff_stddev = float(np.std(finite))
    else:
        diff_median = diff_mad = diff_stddev = np.nan

    n_changed = int(np.count_nonzero(mask_diff))
    summary = {
        "visit": visit,
        "detector": detector,
        "diff_median": diff_median,
        "diff_mad": diff_mad,
        "diff_stddev": diff_stddev,
        "n_mask_pixels_changed": n_changed,
        "frac_mask_pixels_changed": (float(n_changed / mask_diff.size)
                                     if mask_diff.size else 0.0),
    }
    return PixelCompareResult(img1=img1, img2=img2, diff=diff, ratio=ratio,
                              mask_diff=mask_diff, summary=summary)

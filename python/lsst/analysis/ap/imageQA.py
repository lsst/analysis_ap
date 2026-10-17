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
  returns their pixel-wise difference, ratio, mask differences, and a small
  summary.

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

from .imageUtils import get_visit_image


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
    """Return ``{"frac_<plane>": fraction_of_pixels_set}`` for a mask.

    Parameters
    ----------
    mask : `lsst.images.Mask`
        The mask whose planes are to be summed.

    Returns
    -------
    fractions : `dict` [`str`, `float`]
        Per-plane fraction of pixels where that plane bit is set.
    """
    if mask.bbox.area == 0:
        return {}
    return {f"frac_{name}": float(np.mean(mask.get(name))) for name in mask.schema.names}


def _mask_difference(mask1, mask2):
    """Return where two masks differ in any plane.

    Planes are matched by name; a plane absent from one mask counts as unset
    there.

    Parameters
    ----------
    mask1, mask2 : `lsst.images.Mask`
        Masks with the same shape.

    Returns
    -------
    differ : `numpy.ndarray` [`bool`]
        True where at least one plane is set in only one mask.
    """
    differ = np.zeros(mask1.bbox.shape, dtype=bool)
    for name in set(mask1.schema.names) | set(mask2.schema.names):
        set1 = mask1.get(name) if name in mask1.schema.names else False
        set2 = mask2.get(name) if name in mask2.schema.names else False
        differ |= set1 != set2
    return differ


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
        a ``frac_<PLANE>`` entry per mask plane on the image, named as in
        `lsst.images` (e.g. ``frac_SATURATED``).
    """
    diff = get_visit_image(butler, dataset_name, {"visit": visit, "detector": detector},
                           image_type="difference")
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
    img1, img2 : `lsst.images.DifferenceImage`
        The two loaded images, in case the caller wants their masks/WCS.
    diff : `numpy.ndarray`
        Pixel array of ``img1 - img2``.
    ratio : `numpy.ndarray`
        Pixel array of ``img1 / img2``; NaN where ``img2 == 0``.
    mask_diff : `numpy.ndarray` [`bool`]
        True where at least one mask plane, matched by name, is set in only
        one of the two images.
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
    and inspect the residuals. A legacy image is converted to `lsst.images`
    form when read, so it can be compared with an `lsst.images` one.

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
    data_id = {"visit": visit, "detector": detector}
    img1 = get_visit_image(butler1, dataset_name, data_id, image_type="difference")
    img2 = get_visit_image(butler2, dataset_name, data_id, image_type="difference")

    a1 = img1.image.array
    a2 = img2.image.array
    if a1.shape != a2.shape:
        raise ValueError(f"Image shapes differ for visit={visit} detector={detector}: "
                         f"{a1.shape} vs {a2.shape}")

    diff = a1 - a2
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(a2 != 0, a1 / a2, np.nan)

    mask_diff = _mask_difference(img1.mask, img2.mask)

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

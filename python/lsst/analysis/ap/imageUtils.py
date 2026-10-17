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

"""Helpers for images stored either as afw exposures or as `lsst.images`
types.
"""

__all__ = ["as_exposure", "get_exposure"]

import lsst.afw.image as afwImage
from lsst.images import VisitImage
from lsst.images.cells import CellCoadd


def as_exposure(image):
    """Return an image as an afw exposure.

    Parameters
    ----------
    image : `lsst.afw.image.Exposure`, `lsst.images.VisitImage`, or \
            `lsst.images.cells.CellCoadd`
        Image to convert. A `~lsst.images.DifferenceImage` is a
        `~lsst.images.VisitImage`.

    Returns
    -------
    exposure : `lsst.afw.image.Exposure`
        ``image`` itself if it is already an exposure, otherwise its legacy
        conversion, which may share pixels with ``image``.

    Raises
    ------
    TypeError
        Raised if ``image`` is none of the supported types.
    """
    if isinstance(image, afwImage.Exposure):
        return image
    if isinstance(image, (VisitImage, CellCoadd)):
        return image.to_legacy()
    raise TypeError(f"Cannot convert a {type(image).__name__} to an afw exposure.")


def get_exposure(butler, *args, **kwargs):
    """Read an image dataset as an afw exposure, whatever its storage class.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler to read from.
    *args, **kwargs
        Passed to `lsst.daf.butler.Butler.get`.

    Returns
    -------
    exposure : `lsst.afw.image.Exposure`
        The dataset, converted by `as_exposure`.
    """
    return as_exposure(butler.get(*args, **kwargs))

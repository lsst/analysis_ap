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

"""Read AP images as `lsst.images` types, converting legacy afw exposures,
and cut them out on the sky.
"""

__all__ = ["as_masked_image", "as_visit_image", "get_visit_image", "cutout", "cutout_box", "pixel_scale"]

import warnings

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

import lsst.afw.image as afwImage
import lsst.geom
from lsst.daf.butler import DatasetNotFoundError, DatasetRef
from lsst.images import (
    Box,
    DifferenceImage,
    GeneralFrame,
    Image,
    MaskedImage,
    SkyProjection,
    VisitImage,
    get_legacy_difference_image_mask_planes,
    get_legacy_template_mask_planes,
    get_legacy_visit_image_mask_planes,
)
from lsst.images.fits import parse_legacy_bunit

_IMAGE_TYPES = {"science": VisitImage, "template": DifferenceImage, "difference": DifferenceImage}
"""`lsst.images` type of each AP image role."""

_PLANE_MAPS = {
    "science": get_legacy_visit_image_mask_planes,
    "template": get_legacy_template_mask_planes,
    "difference": get_legacy_difference_image_mask_planes,
}
"""Legacy mask-plane map factory of each AP image role."""


def _legacy_unit(exposure):
    """Return the pixel unit of a legacy exposure.

    ``BUNIT`` is used if set. Otherwise an exposure whose photometric
    calibration is the identity is in nJy, and any other is instrumental,
    which ``from_legacy`` also calls electrons.
    """
    if bunit := exposure.metadata.get("BUNIT"):
        return parse_legacy_bunit(bunit, instrumental_unit=u.electron)
    photo_calib = exposure.photoCalib
    if photo_calib is not None and photo_calib.getCalibrationMean() == 1.0:
        return u.nJy
    return u.electron


def as_visit_image(image, *, image_type="science", exposure_record=None, detector=None):
    """Return an AP image as an `lsst.images` type.

    Parameters
    ----------
    image : `lsst.images.VisitImage` or `lsst.afw.image.Exposure`
        Image to convert. An `lsst.images.VisitImage` (which includes
        `~lsst.images.DifferenceImage`) is returned unchanged.
    image_type : {"science", "template", "difference"}, optional
        Role of a legacy image, which sets the type it becomes (a
        `~lsst.images.VisitImage` for a science image, a
        `~lsst.images.DifferenceImage` otherwise) and the mask planes it
        keeps.
    exposure_record : `lsst.daf.butler.DimensionRecord`, optional
        ``exposure`` dimension record of a legacy image. Without it the
        converted observation information lacks the fields the record
        supplies.
    detector : `lsst.afw.cameraGeom.Detector` or \
            `lsst.images.cameras.Detector`, optional
        Detector to attach to a legacy image that has none, such as a
        legacy template; the science image's detector is the right one.

    Returns
    -------
    image : `lsst.images.VisitImage`
        The image in `lsst.images` form. A converted image shares its image
        and variance pixels with ``image``.

    Raises
    ------
    TypeError
        Raised if ``image`` is neither an `lsst.images.VisitImage` nor an
        `lsst.afw.image.Exposure`.
    ValueError
        Raised if a legacy image has no detector and none is given.
    """
    if isinstance(image, VisitImage):
        return image
    if not isinstance(image, afwImage.Exposure):
        raise TypeError(f"Cannot convert a {type(image).__name__} to an lsst.images VisitImage.")
    # Conversion edits the exposure's metadata, so convert a shallow copy
    # with its own metadata.
    exposure = type(image)(image, deep=False)
    exposure.setMetadata(image.getMetadata().deepCopy())
    if exposure.getDetector() is None:
        if detector is None:
            raise ValueError(f"The legacy {image_type} image has no detector; pass the science image's "
                             "detector to convert it.")
        exposure.setDetector(detector if hasattr(detector, "getBBox") else detector.to_legacy())
    with warnings.catch_warnings():
        # The analysis tools do not use the fields a missing record leaves
        # empty.
        warnings.filterwarnings("ignore", message="Converting without an exposure dimension record")
        return _IMAGE_TYPES[image_type].from_legacy(exposure, exposure_record=exposure_record,
                                                    unit=_legacy_unit(image),
                                                    plane_map=_PLANE_MAPS[image_type]())


def as_masked_image(image, *, image_type="science"):
    """Return the pixels of an AP image as an `lsst.images.MaskedImage`.

    Use this for images without the detector, PSF and other components a
    `~lsst.images.VisitImage` needs, such as a post-ISR image.

    Parameters
    ----------
    image : `lsst.images.MaskedImage` or `lsst.afw.image.Exposure`
        Image to convert. An `lsst.images.MaskedImage` (which includes
        `~lsst.images.VisitImage`) is returned unchanged.
    image_type : {"science", "template", "difference"}, optional
        Role of a legacy image, which sets the mask planes it keeps.

    Returns
    -------
    image : `lsst.images.MaskedImage`
        The image, mask and variance, with a sky projection if ``image`` has
        a WCS. A converted image shares its image and variance pixels with
        ``image``.

    Raises
    ------
    TypeError
        Raised if ``image`` is neither an `lsst.images.MaskedImage` nor an
        `lsst.afw.image.Exposure`.
    """
    if isinstance(image, MaskedImage):
        return image
    if not isinstance(image, afwImage.Exposure):
        raise TypeError(f"Cannot convert a {type(image).__name__} to an lsst.images MaskedImage.")
    sky_projection = None
    if image.wcs is not None:
        sky_projection = SkyProjection.from_legacy(image.wcs, GeneralFrame(unit=u.pix))
    return MaskedImage.from_legacy(image.maskedImage, unit=_legacy_unit(image),
                                   plane_map=_PLANE_MAPS[image_type](), sky_projection=sky_projection)


def _exposure_record(butler, data_id):
    """Return the ``exposure`` dimension record of a visit, or `None`.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler to query.
    data_id : `lsst.daf.butler.DataCoordinate`
        Data ID that includes the visit.
    """
    records = list(butler.query_dimension_records("exposure", data_id=data_id, explain=False))
    return records[0] if records else None


def get_visit_image(butler, dataset, data_id=None, *, image_type="science", detector=None, **kwargs):
    """Read an AP image as an `lsst.images` type, whatever its storage
    class.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler to read from.
    dataset : `str` or `lsst.daf.butler.DatasetRef`
        Dataset type name, or a resolved reference.
    data_id : `dict`, optional
        Data ID of the image; ignored when ``dataset`` is a reference.
    image_type : {"science", "template", "difference"}, optional
        Role of the image; see `as_visit_image`.
    detector : `lsst.afw.cameraGeom.Detector` or \
            `lsst.images.cameras.Detector`, optional
        Detector for a legacy image that has none; see `as_visit_image`.
    **kwargs
        Additional data ID keys, passed to
        `lsst.daf.butler.Butler.find_dataset`.

    Returns
    -------
    image : `lsst.images.VisitImage`
        The image, converted with the ``exposure`` record of its visit if
        it is stored as a legacy exposure.

    Raises
    ------
    lsst.daf.butler.DatasetNotFoundError
        Raised if the butler has no such dataset.
    """
    if isinstance(dataset, DatasetRef):
        ref = dataset
    else:
        ref = butler.find_dataset(dataset, data_id, **kwargs)
        if ref is None:
            raise DatasetNotFoundError(f"No {dataset!r} dataset with data ID {data_id}.")
    image = butler.get(ref)
    if isinstance(image, VisitImage):
        return image
    return as_visit_image(image, image_type=image_type, exposure_record=_exposure_record(butler, ref.dataId),
                          detector=detector)


def _to_sky_coord(center):
    """Return a sky position as an `astropy.coordinates.SkyCoord`."""
    if isinstance(center, lsst.geom.SpherePoint):
        return SkyCoord(center.getRa().asDegrees(), center.getDec().asDegrees(), unit="deg")
    return center


def cutout_box(image, bbox):
    """Copy the pixels of an image within a box.

    Parameters
    ----------
    image : `lsst.images.MaskedImage`
        Image to cut out of.
    bbox : `lsst.images.Box`
        Box to copy, which may extend past the edges of ``image``.

    Returns
    -------
    cutout : `lsst.images.Image`
        Copy of the image plane within ``bbox``; pixels outside ``image``
        are NaN.

    Raises
    ------
    lsst.images.NoOverlapError
        Raised if ``bbox`` does not overlap ``image``.
    """
    result = Image(np.nan, bbox=bbox, dtype=image.image.array.dtype, unit=image.unit,
                   sky_projection=image.sky_projection)
    overlap = bbox.intersection(image.bbox)
    result[overlap].array[...] = image.image[overlap].array
    return result


def cutout(image, center, size):
    """Copy a square of pixels centered on a sky position.

    Parameters
    ----------
    image : `lsst.images.MaskedImage`
        Image to cut out of.
    center : `astropy.coordinates.SkyCoord` or `lsst.geom.SpherePoint`
        Sky position to center the cutout on.
    size : `int`
        Width and height of the cutout, in pixels.

    Returns
    -------
    cutout : `lsst.images.Image`
        Copy of the image plane in the square; pixels outside ``image`` are
        NaN. The square is placed as by `lsst.afw.image.Exposure.getCutout`.

    Raises
    ------
    lsst.images.NoOverlapError
        Raised if the square does not overlap ``image``.
    """
    pixel = image.sky_projection.sky_to_pixel(_to_sky_coord(center))
    legacy_box = lsst.geom.Box2I.makeCenteredBox(lsst.geom.Point2D(float(pixel.x), float(pixel.y)),
                                                 lsst.geom.Extent2I(size, size))
    return cutout_box(image, Box.from_legacy(legacy_box))


def pixel_scale(image):
    """Return the pixel scale at the center of an image.

    Parameters
    ----------
    image : `lsst.images.MaskedImage`
        Image with a sky projection.

    Returns
    -------
    scale : `float`
        Square root of the pixel area on the sky, in arcseconds.
    """
    x = 0.5*(image.bbox.x.start + image.bbox.x.stop - 1)
    y = 0.5*(image.bbox.y.start + image.bbox.y.stop - 1)
    sky = image.sky_projection.pixel_to_sky(x=np.array([x, x + 1.0, x]), y=np.array([y, y, y + 1.0]))
    dx = sky[0].spherical_offsets_to(sky[1])
    dy = sky[0].spherical_offsets_to(sky[2])
    area = abs(dx[0]*dy[1] - dx[1]*dy[0])
    return float(np.sqrt(area.to_value(u.arcsec**2)))


def _sky_to_pixel(wcs, ra, dec):
    """Transform sky positions in degrees to pixels.

    Parameters
    ----------
    wcs : `lsst.images.SkyProjection` or `lsst.afw.geom.SkyWcs`
        Transform to apply.
    ra, dec : array-like
        Sky positions, in degrees.

    Returns
    -------
    x, y : `numpy.ndarray`
        Pixel positions.
    """
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    if hasattr(wcs, "sky_to_pixel"):
        pixels = wcs.sky_to_pixel(SkyCoord(ra, dec, unit="deg"))
        return np.asarray(pixels.x, dtype=float), np.asarray(pixels.y, dtype=float)
    return wcs.skyToPixelArray(ra, dec, degrees=True)


def _pixel_to_sky(wcs, x, y):
    """Transform pixel positions to sky positions in degrees.

    Parameters
    ----------
    wcs : `lsst.images.SkyProjection` or `lsst.afw.geom.SkyWcs`
        Transform to apply.
    x, y : array-like
        Pixel positions.

    Returns
    -------
    ra, dec : `numpy.ndarray`
        Sky positions, in degrees.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if hasattr(wcs, "pixel_to_sky"):
        sky = wcs.pixel_to_sky(x=x, y=y)
        return np.asarray(sky.ra.deg), np.asarray(sky.dec.deg)
    return wcs.pixelToSkyArray(x, y, degrees=True)


def _box_edges(bbox):
    """Return the outer pixel edges of a box, ``(x_min, x_max, y_min,
    y_max)``.

    Parameters
    ----------
    bbox : `lsst.images.Box` or `lsst.geom.Box2I`
        Integer pixel box.
    """
    if isinstance(bbox, lsst.geom.Box2I):
        bbox = Box.from_legacy(bbox)
    return (bbox.x.start - 0.5, bbox.x.stop - 0.5, bbox.y.start - 0.5, bbox.y.stop - 0.5)

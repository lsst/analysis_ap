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

"""Tests of analysis_ap functions reading images in the `lsst.images`
format.
"""

import unittest
import uuid
from unittest import mock

import astropy.time
import astropy.units as u
import numpy as np
import PIL

import lsst.afw.cameraGeom.testUtils as cameraGeomTestUtils
import lsst.afw.detection as afwDetection
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.daf.base as dafBase
import lsst.geom
import lsst.utils.tests
from lsst.analysis.ap import imageQA, nb_utils, plotImageSubtractionCutouts, ppdb
from lsst.analysis.ap.imageUtils import as_exposure, get_exposure
from lsst.daf.butler import DimensionUniverse, Timespan
from lsst.images import Box, DifferenceImage, DifferenceImageTemplateInfo, Polygon, VisitImage

SKY_CENTER = lsst.geom.SpherePoint(45.0, 30.0, lsst.geom.degrees)


def make_exposure(seed=1):
    """Make a small legacy exposure that `lsst.images` can convert.

    The detector comes from `~lsst.afw.cameraGeom.testUtils.CameraWrapper`,
    since conversion needs amplifier raw geometry and a field-angle
    transform.
    """
    detector = list(cameraGeomTestUtils.CameraWrapper().camera)[0]
    bbox = detector.getBBox()
    exposure = afwImage.ExposureF(bbox)
    rng = np.random.default_rng(seed)
    exposure.image.array[:] = rng.normal(scale=10.0, size=exposure.image.array.shape)
    exposure.variance.array[:] = 100.0
    exposure.mask.array[10:20, 10:30] = exposure.mask.getPlaneBitMask("BAD")
    exposure.mask.array[50:55, 60:62] = exposure.mask.getPlaneBitMask("DETECTED")
    exposure.setDetector(detector)
    exposure.setWcs(afwGeom.makeSkyWcs(lsst.geom.Point2D(bbox.getCenter()), SKY_CENTER,
                                       afwGeom.makeCdMatrix(0.2*lsst.geom.arcseconds)))
    exposure.setPsf(afwDetection.GaussianPsf(21, 21, 2.0))
    exposure.setFilter(afwImage.FilterLabel(band="g", physical="g_test"))
    exposure.setPhotoCalib(afwImage.PhotoCalib(1.0))
    exposure.info.setVisitInfo(afwImage.VisitInfo(
        id=12345, exposureTime=30.0,
        date=dafBase.DateTime(60815.000174, dafBase.DateTime.MJD, dafBase.DateTime.TAI)))
    return exposure


def make_exposure_record():
    """Make the ``exposure`` dimension record that conversion needs."""
    begin = astropy.time.Time(60815.0, format="mjd", scale="tai")
    return DimensionUniverse()["exposure"].RecordClass(
        instrument="testCam", id=12345, obs_id="test_12345", group="12345", day_obs=20250520,
        physical_filter="g_test", exposure_time=30.0, seq_num=1, seq_start=1, seq_end=1,
        can_see_sky=True, azimuth=120.0, zenith_angle=30.0,
        timespan=Timespan(begin=begin, end=begin + astropy.time.TimeDelta(30.0, format="sec")),
    )


def to_native(exposure, image_type=DifferenceImage):
    """Convert a legacy exposure to an `lsst.images` type."""
    return image_type.from_legacy(exposure, exposure_record=make_exposure_record(), unit=u.nJy)


class FakeButler:
    """Butler stand-in that returns fixed objects by dataset name."""

    def __init__(self, datasets):
        self.datasets = datasets

    def get(self, name, *args, **kwargs):
        return self.datasets[name]


class AsExposureTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.exposure = make_exposure()

    def testLegacy(self):
        self.assertIs(as_exposure(self.exposure), self.exposure)

    def testNative(self):
        for image_type in (VisitImage, DifferenceImage):
            with self.subTest(image_type=image_type.__name__):
                converted = as_exposure(to_native(self.exposure, image_type))
                self.assertIsInstance(converted, afwImage.ExposureF)
                self.assertEqual(converted.getBBox(), self.exposure.getBBox())
                self.assertImagesEqual(converted.image, self.exposure.image)
                self.assertImagesEqual(converted.variance, self.exposure.variance)
                self.assertMasksEqual(converted.mask, self.exposure.mask)
                self.assertEqual(converted.filter, self.exposure.filter)
                self.assertIsNotNone(converted.wcs)

    def testUnsupported(self):
        with self.assertRaises(TypeError):
            as_exposure(self.exposure.maskedImage)

    def testGetExposure(self):
        butler = FakeButler({"difference_image": to_native(self.exposure)})
        self.assertIsInstance(get_exposure(butler, "difference_image", {"visit": 1}), afwImage.ExposureF)


class NativeImageFunctionsTestCase(lsst.utils.tests.TestCase):
    """Test that functions give the same result for a legacy image and its
    `lsst.images` conversion.
    """

    def setUp(self):
        self.legacy = make_exposure()
        self.native = to_native(self.legacy)

    def testImageDiffStats(self):
        legacy = imageQA.image_diff_stats(FakeButler({"difference_image": self.legacy}), 1, 2)
        native = imageQA.image_diff_stats(FakeButler({"difference_image": self.native}), 1, 2)
        self.assertEqual(native, legacy)
        self.assertEqual(legacy["frac_BAD"], 200/self.legacy.image.array.size)
        self.assertEqual(legacy["frac_DETECTED"], 10/self.legacy.image.array.size)

    def testPixelCompare(self):
        result = imageQA.pixel_compare(FakeButler({"difference_image": self.legacy}),
                                       FakeButler({"difference_image": self.native}), 1, 2)
        self.assertEqual(result.summary["diff_stddev"], 0.0)
        self.assertEqual(result.summary["n_mask_pixels_changed"], 0)

    def testRegionFromExposure(self):
        legacy = ppdb.region_from_exposure(self.legacy)
        native = ppdb.region_from_exposure(self.native)
        np.testing.assert_allclose(native, legacy, rtol=0, atol=1e-12)

    def testGenerateImage(self):
        task = plotImageSubtractionCutouts.PlotImageSubtractionCutoutsTask(output_path="")
        cutout = task.generate_image(self.native, self.native, self.native, SKY_CENTER, 0.2)
        with PIL.Image.open(cutout) as image:
            self.assertGreater(image.width, 0)

    def testButlerCache(self):
        config = plotImageSubtractionCutouts.PlotImageSubtractionCutoutsConfig()
        config.diff_image_type = None
        butler = FakeButler({"preliminary_visit_image": to_native(self.legacy, VisitImage),
                             "template_detector": self.native,
                             "difference_image": self.native})
        plotImageSubtractionCutouts.butler_cache.set(butler, config)
        for exposure in plotImageSubtractionCutouts.butler_cache.get_exposures("testCam", 2, 1):
            self.assertIsInstance(exposure, afwImage.ExposureF)


class TemplatePatchOutlinesTestCase(lsst.utils.tests.TestCase):
    """Test finding the coadds of a template stored as an
    `lsst.images.DifferenceImage`.
    """

    def setUp(self):
        self.wcs = afwGeom.makeSkyWcs(lsst.geom.Point2D(0.0, 0.0), SKY_CENTER,
                                      afwGeom.makeCdMatrix(0.2*lsst.geom.arcseconds))
        self.ids = [uuid.uuid4(), uuid.uuid4()]
        # Listed out of order, to check that the outlines are sorted.
        self.templates = [
            DifferenceImageTemplateInfo(
                skymap="test", tract=1, patch=patch, dataset_id=dataset_id, dataset_run="run",
                bounds=Polygon.from_box(Box.from_shape((100, 60), start=(0, x0))),
                psf_shape_xx=1.0, psf_shape_yy=1.0, psf_shape_xy=0.0, psf_shape_flag=False,
            )
            for patch, dataset_id, x0 in [(1, self.ids[1], 40), (0, self.ids[0], 0)]
        ]
        # The coadd of patch 0 shares the difference image's pixel grid.
        self.coadd_bbox = lsst.geom.Box2I(lsst.geom.Point2I(-10, -20), lsst.geom.Extent2I(70, 140))
        self.ref = mock.MagicMock()
        self.ref.datasetType.storageClass.allComponents.return_value = {"wcs": None, "bbox": None}
        self.ref.makeComponentRef.side_effect = lambda component: component

        def get(target, *args, storageClass=None):
            return {"template_detector.templates": self.templates, "wcs": self.wcs,
                    "bbox": self.coadd_bbox}[target]

        self.butler = mock.MagicMock()
        self.butler.get_dataset_type.return_value.storageClass.allComponents.return_value = {
            "templates": None}
        self.butler.get.side_effect = get
        # The coadd of patch 1 is missing from the butler.
        self.butler.get_dataset.side_effect = {self.ids[0]: self.ref, self.ids[1]: None}.get

    def testOutlines(self):
        outlines, refs = nb_utils._template_patch_outlines(
            self.butler, "template_detector", {"visit": 1, "detector": 2}, self.wcs,
            coadd_dataset="unused", band="g")
        self.assertEqual(list(outlines), [(1, 0), (1, 1)])
        self.assertEqual(refs, {(1, 0): self.ref})
        # A found coadd is outlined by its own bounding box.
        expected = lsst.geom.Box2D(self.coadd_bbox).getCorners()
        np.testing.assert_allclose(outlines[(1, 0)], [(p.getX(), p.getY()) for p in expected], atol=1e-6)
        # A missing coadd is outlined by its recorded overlap.
        corners = outlines[(1, 1)]
        self.assertEqual(len(corners), 4)
        self.assertEqual({x for x, _ in corners}, {39.5, 99.5})
        self.assertEqual({y for _, y in corners}, {-0.5, 99.5})


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

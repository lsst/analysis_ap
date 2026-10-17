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

"""Tests of reading AP images as `lsst.images` types, and of the functions
that use them.
"""

import collections
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
from lsst.analysis.ap.imageUtils import (
    _pixel_to_sky, _sky_to_pixel, as_masked_image, as_visit_image, cutout, cutout_box, get_visit_image,
    pixel_scale,
)
from lsst.daf.butler import DatasetNotFoundError, DimensionUniverse, Timespan
from lsst.images import (
    Box, DifferenceImage, DifferenceImageTemplateInfo, MaskedImage, NoOverlapError, Polygon, VisitImage,
)

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
    """Make the ``exposure`` dimension record of the test exposure."""
    begin = astropy.time.Time(60815.0, format="mjd", scale="tai")
    return DimensionUniverse()["exposure"].RecordClass(
        instrument="testCam", id=12345, obs_id="test_12345", group="12345", day_obs=20250520,
        physical_filter="g_test", exposure_time=30.0, seq_num=1, seq_start=1, seq_end=1,
        can_see_sky=True, azimuth=120.0, zenith_angle=30.0,
        timespan=Timespan(begin=begin, end=begin + astropy.time.TimeDelta(30.0, format="sec")),
    )


FakeRef = collections.namedtuple("FakeRef", ["name", "dataId"])


class FakeButler:
    """Butler stand-in that returns fixed objects by dataset name.

    Component reads (``"<dataset>.<component>"``) return that attribute of
    the legacy exposure or `lsst.images` image stored under ``<dataset>``.
    """

    def __init__(self, datasets, record=None):
        self.datasets = datasets
        self.record = record
        self.record_queries = []

    def find_dataset(self, name, data_id=None, **kwargs):
        return FakeRef(name, data_id) if name in self.datasets else None

    def get(self, target, *args, **kwargs):
        name = target.name if isinstance(target, FakeRef) else target
        if "." in name:
            parent, component = name.split(".")
            dataset = self.datasets[parent]
            if isinstance(dataset, afwImage.Exposure):
                return {"detector": dataset.getDetector, "bbox": dataset.getBBox}[component]()
            return getattr(dataset, component)
        return self.datasets[name]

    def query_dimension_records(self, element, data_id=None, explain=True):
        self.record_queries.append(data_id)
        return [self.record] if self.record is not None else []


class AsVisitImageTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.exposure = make_exposure()

    def testNativePassesThrough(self):
        image = as_visit_image(self.exposure, image_type="difference")
        self.assertIs(as_visit_image(image), image)
        self.assertIs(as_masked_image(image), image)

    def testRoles(self):
        for image_type, image_class in [("science", VisitImage), ("template", DifferenceImage),
                                        ("difference", DifferenceImage)]:
            with self.subTest(image_type=image_type):
                image = as_visit_image(self.exposure, image_type=image_type,
                                       exposure_record=make_exposure_record())
                self.assertIs(type(image), image_class)
                self.assertEqual(image.bbox, Box.from_legacy(self.exposure.getBBox()))
                np.testing.assert_array_equal(image.image.array, self.exposure.image.array)
                np.testing.assert_array_equal(image.variance.array, self.exposure.variance.array)
                bad = self.exposure.mask.array == self.exposure.mask.getPlaneBitMask("BAD")
                np.testing.assert_array_equal(image.mask.get("BAD"), bad)
                self.assertEqual(image.band, "g")

    def testUnits(self):
        self.assertEqual(as_visit_image(self.exposure).unit, u.nJy)
        self.exposure.setPhotoCalib(afwImage.PhotoCalib(57.5))
        image = as_visit_image(self.exposure)
        self.assertEqual(image.unit, u.electron)
        self.assertIsNotNone(image.photometric_scaling)
        self.exposure.metadata["BUNIT"] = "nJy"
        self.assertEqual(as_visit_image(self.exposure).unit, u.nJy)
        # The caller's metadata is left alone, so a second conversion works.
        self.assertEqual(self.exposure.metadata["BUNIT"], "nJy")
        self.assertEqual(as_visit_image(self.exposure).unit, u.nJy)

    def testTemplateDetector(self):
        detector = self.exposure.getDetector()
        self.exposure.setDetector(None)
        with self.assertRaisesRegex(ValueError, "no detector"):
            as_visit_image(self.exposure, image_type="template")
        native_detector = as_visit_image(make_exposure()).detector
        for given in (detector, native_detector):
            image = as_visit_image(self.exposure, image_type="template", detector=given)
            self.assertEqual(image.detector.to_legacy().getId(), detector.getId())
        self.assertIsNone(self.exposure.getDetector())

    def testUnsupported(self):
        with self.assertRaises(TypeError):
            as_visit_image(self.exposure.maskedImage)
        with self.assertRaises(TypeError):
            as_masked_image(self.exposure.maskedImage)

    def testMaskedImage(self):
        # A post-ISR image has no PSF or WCS.
        self.exposure.setPsf(None)
        self.exposure.setWcs(None)
        image = as_masked_image(self.exposure)
        self.assertIs(type(image), MaskedImage)
        np.testing.assert_array_equal(image.image.array, self.exposure.image.array)
        self.assertEqual(int(image.mask.get("BAD").sum()), 200)


class GetVisitImageTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.exposure = make_exposure()
        self.record = make_exposure_record()

    def testLegacy(self):
        butler = FakeButler({"difference_image": self.exposure}, record=self.record)
        image = get_visit_image(butler, "difference_image", {"visit": 12345}, image_type="difference")
        self.assertIsInstance(image, DifferenceImage)
        self.assertEqual(butler.record_queries, [{"visit": 12345}])
        self.assertEqual(image.obs_info.exposure_time, 30.0*u.s)

    def testNative(self):
        native = as_visit_image(self.exposure, image_type="difference")
        butler = FakeButler({"difference_image": native})
        self.assertIs(get_visit_image(butler, "difference_image", {"visit": 1}), native)
        self.assertEqual(butler.record_queries, [])

    def testMissing(self):
        with self.assertRaises(DatasetNotFoundError):
            get_visit_image(FakeButler({}), "difference_image", {"visit": 1})


class CutoutTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        self.exposure = make_exposure()
        self.image = as_visit_image(self.exposure)

    def testMatchesLegacyCutout(self):
        # Inside the image, and across a corner, where the cutout is padded.
        for x, y in [(100.3, 80.7), (101.0, 60.0), (1.2, 2.6)]:
            with self.subTest(x=x, y=y):
                center = self.exposure.wcs.pixelToSky(x, y)
                legacy = self.exposure.getCutout(center, lsst.geom.Extent2I(9, 9))
                stamp = cutout(self.image, center, 9)
                self.assertEqual(stamp.bbox, Box.from_legacy(legacy.getBBox()))
                np.testing.assert_array_equal(stamp.array, legacy.image.array)

    def testOffImage(self):
        far = self.exposure.wcs.pixelToSky(-500.0, -500.0)
        with self.assertRaises(NoOverlapError):
            cutout(self.image, far, 9)

    def testCutoutBox(self):
        box = Box.from_legacy(lsst.geom.Box2I(lsst.geom.Point2I(5, 7), lsst.geom.Extent2I(10, 12)))
        np.testing.assert_array_equal(cutout_box(self.image, box).array,
                                      self.exposure.image[box.to_legacy()].array)

    def testPixelScale(self):
        expected = self.exposure.wcs.getPixelScale(self.exposure.getBBox().getCenter()).asArcseconds()
        self.assertAlmostEqual(pixel_scale(self.image), expected, places=9)

    def testSkyPixelTransforms(self):
        x, y = np.array([3.0, 150.5]), np.array([9.0, 40.25])
        ra, dec = _pixel_to_sky(self.image.sky_projection, x, y)
        np.testing.assert_allclose((ra, dec), _pixel_to_sky(self.exposure.wcs, x, y), rtol=0, atol=1e-10)
        for wcs in (self.image.sky_projection, self.exposure.wcs):
            np.testing.assert_allclose(_sky_to_pixel(wcs, ra, dec), (x, y), rtol=0, atol=1e-6)


class NativeImageFunctionsTestCase(lsst.utils.tests.TestCase):
    """Test that functions give the same result for a legacy image and its
    `lsst.images` form.
    """

    def setUp(self):
        self.legacy = make_exposure()
        self.native = as_visit_image(self.legacy, image_type="difference")

    def testImageDiffStats(self):
        legacy = imageQA.image_diff_stats(FakeButler({"difference_image": self.legacy}), 1, 2)
        native = imageQA.image_diff_stats(FakeButler({"difference_image": self.native}), 1, 2)
        self.assertEqual(native, legacy)
        n_pixels = self.legacy.image.array.size
        self.assertEqual(legacy["frac_BAD"], 200/n_pixels)
        self.assertEqual(legacy["frac_DETECTED"], 10/n_pixels)

    def testPixelCompare(self):
        result = imageQA.pixel_compare(FakeButler({"difference_image": self.legacy}),
                                       FakeButler({"difference_image": self.native}), 1, 2)
        self.assertEqual(result.summary["diff_stddev"], 0.0)
        self.assertEqual(result.summary["n_mask_pixels_changed"], 0)

    def testMaskDifference(self):
        other = make_exposure()
        other.mask.array[0, 0] |= other.mask.getPlaneBitMask("SAT")
        other = as_visit_image(other)
        differ = imageQA._mask_difference(self.native.mask, other.mask)
        self.assertEqual(int(differ.sum()), 1)
        self.assertTrue(differ[0, 0])

    def testRegionFromExposure(self):
        bbox = self.legacy.getBBox()
        center = self.legacy.wcs.pixelToSky(bbox.getCenter())
        radius = max(center.separation(self.legacy.wcs.pixelToSky(lsst.geom.Point2D(corner)))
                     for corner in bbox.getCorners()) + 5*lsst.geom.arcseconds
        expected = (center.getRa().asDegrees(), center.getDec().asDegrees(), radius.asDegrees())
        for image in (self.native, self.legacy):
            np.testing.assert_allclose(ppdb.region_from_exposure(image, padding=5.0), expected,
                                       rtol=0, atol=1e-10)

    def testGenerateImage(self):
        task = plotImageSubtractionCutouts.PlotImageSubtractionCutoutsTask(output_path="")
        legacy_template = make_exposure(seed=2)
        legacy_template.setDetector(None)
        # A legacy template is given the science image's detector.
        stamp = task.generate_image(self.legacy, legacy_template, self.native, SKY_CENTER, 0.2)
        with PIL.Image.open(stamp) as image:
            self.assertGreater(image.width, 0)

    def testButlerCache(self):
        config = plotImageSubtractionCutouts.PlotImageSubtractionCutoutsConfig()
        config.diff_image_type = None
        legacy_template = make_exposure(seed=2)
        legacy_template.setDetector(None)
        butler = FakeButler({"preliminary_visit_image": self.legacy,
                             "template_detector": legacy_template,
                             "difference_image": self.native})
        plotImageSubtractionCutouts.butler_cache.set(butler, config)
        science, template, difference = plotImageSubtractionCutouts.butler_cache.get_images("testCam", 2, 1)
        self.assertIs(type(science), VisitImage)
        self.assertIs(type(template), DifferenceImage)
        self.assertIs(difference, self.native)

    def testLoadCutout(self):
        butler = FakeButler({"preliminary_visit_image": self.legacy, "difference_image": self.native})
        row = mock.Mock(visit=1, detector=2, ra=SKY_CENTER.getRa().asDegrees(),
                        dec=SKY_CENTER.getDec().asDegrees())
        data = nb_utils._load_cutout(butler, row, size=15, image_type="difference",
                                     image_datasets=nb_utils._IMAGE_DATASETS)
        legacy = self.legacy.getCutout(SKY_CENTER, lsst.geom.Extent2I(15, 15))
        np.testing.assert_array_equal(data["data"], legacy.image.array)
        self.assertEqual((data["x0"], data["y0"]), (legacy.getX0(), legacy.getY0()))


class TemplatePatchOutlinesTestCase(lsst.utils.tests.TestCase):
    """Test finding the coadds of a template stored as an
    `lsst.images.DifferenceImage`.
    """

    def setUp(self):
        exposure = make_exposure()
        # The coadd of patch 0 shares the difference image's pixel grid.
        self.coadd_wcs = exposure.wcs
        self.sky_projection = as_visit_image(exposure).sky_projection
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
        self.coadd_bbox = lsst.geom.Box2I(lsst.geom.Point2I(-10, -20), lsst.geom.Extent2I(70, 140))
        self.ref = mock.MagicMock()
        self.ref.datasetType.storageClass.allComponents.return_value = {"wcs": None, "bbox": None}
        self.ref.makeComponentRef.side_effect = lambda component: component

        def get(target, *args, storageClass=None):
            return {"template_detector.templates": self.templates, "wcs": self.coadd_wcs,
                    "bbox": self.coadd_bbox}[target]

        self.butler = mock.MagicMock()
        self.butler.get_dataset_type.return_value.storageClass.allComponents.return_value = {
            "templates": None}
        self.butler.get.side_effect = get
        # The coadd of patch 1 is missing from the butler.
        self.butler.get_dataset.side_effect = {self.ids[0]: self.ref, self.ids[1]: None}.get

    def testOutlines(self):
        outlines, refs = nb_utils._template_patch_outlines(
            self.butler, "template_detector", {"visit": 1, "detector": 2}, self.sky_projection,
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

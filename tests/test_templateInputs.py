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

import types
import unittest
import uuid

import astropy.table
import numpy as np

import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.geom
import lsst.utils.tests
from lsst.analysis.ap.templateInputs import (
    _CellCoaddReader, _combine_weights, _LegacyCoaddReader, get_template_inputs, summarize_template_inputs,
)
from lsst.daf.base import DateTime
from lsst.images import YX, Box, DifferenceImageTemplateInfo, Polygon, TractFrame
from lsst.images.cells import CellGrid, CoaddProvenance
from lsst.images.tests import make_random_sky_projection


def _make_wcs(ra=150.0, dec=2.0):
    """Return a TAN WCS with 0.2 arcsec pixels and its reference pixel at
    the origin.
    """
    return afwGeom.makeSkyWcs(
        crpix=lsst.geom.Point2D(0.0, 0.0),
        crval=lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees),
        cdMatrix=afwGeom.makeCdMatrix(scale=0.2*lsst.geom.arcseconds),
    )


def _make_ccds(wcs, rows):
    """Return a legacy coadd ``ccds`` catalog.

    Parameters
    ----------
    wcs : `lsst.afw.geom.SkyWcs`
        WCS shared by every input.
    rows : `list` [`tuple`]
        ``(visit, ccd, weight, bbox, valid_box, mjd)`` for each input.
    """
    schema = afwTable.ExposureTable.makeMinimalSchema()
    schema.addField("visit", type="L", doc="")
    schema.addField("ccd", type="I", doc="")
    schema.addField("weight", type="D", doc="")
    ccds = afwTable.ExposureCatalog(schema)
    for visit, ccd, weight, bbox, valid_box, mjd in rows:
        record = ccds.addNew()
        record["visit"] = visit
        record["ccd"] = ccd
        record["weight"] = weight
        record.setWcs(wcs)
        record.setBBox(bbox)
        record.setValidPolygon(afwGeom.Polygon(lsst.geom.Box2D(valid_box)))
        record.setVisitInfo(afwImage.VisitInfo(exposureTime=30.0,
                                               date=DateTime(mjd, DateTime.MJD, DateTime.TAI)))
    return ccds


def _sky(wcs, x, y):
    """Return the (ra, dec) arrays, in degrees, of pixel positions."""
    points = [wcs.pixelToSky(lsst.geom.Point2D(px, py)) for px, py in zip(x, y)]
    return (np.array([p.getRa().asDegrees() for p in points]),
            np.array([p.getDec().asDegrees() for p in points]))


class CombineWeightsTestCase(lsst.utils.tests.TestCase):
    """Test the patch and tract weighting of overlapping coadds."""

    def _inputs(self, weights):
        return astropy.table.Table({"coadd_weight": np.array(weights, dtype=float)})

    def testSingleCoadd(self):
        fractions = _combine_weights([("a", 1, self._inputs([1.0, 3.0]))], {})
        self.assertEqual(fractions, {"a": 0.25})

    def testPatchOverlap(self):
        # Patch weights go as variance**-0.5: 1/1 and 1/2, so 2/3 and 1/3.
        coadds = [("a", 1, self._inputs([2.0])), ("b", 1, self._inputs([1.0, 1.0]))]
        fractions = _combine_weights(coadds, {"a": 1.0, "b": 4.0})
        self.assertAlmostEqual(fractions["a"]*2.0, 2.0/3.0)
        self.assertAlmostEqual(fractions["b"]*2.0, 1.0/3.0)

    def testTractOverlap(self):
        # Tract 1 averages variances 1 and 4 with weights 2/3 and 1/3, giving
        # 2; tract 2 has variance 8. Tract weights 2**-0.5 and 8**-0.5 are in
        # the ratio 2:1.
        coadds = [("a", 1, self._inputs([1.0])), ("b", 1, self._inputs([1.0])),
                  ("c", 2, self._inputs([1.0]))]
        fractions = _combine_weights(coadds, {"a": 1.0, "b": 4.0, "c": 8.0})
        self.assertAlmostEqual(fractions["a"], 2.0/3.0*2.0/3.0)
        self.assertAlmostEqual(fractions["b"], 2.0/3.0*1.0/3.0)
        self.assertAlmostEqual(fractions["c"], 1.0/3.0)
        self.assertAlmostEqual(sum(fractions.values()), 1.0)

    def testUnusableCoadds(self):
        # A coadd with no inputs, or with no valid variance, is left out the
        # way GetTemplateTask leaves out its pixels.
        coadds = [("a", 1, self._inputs([1.0])), ("b", 1, self._inputs([1.0])),
                  ("c", 1, self._inputs([]))]
        fractions = _combine_weights(coadds, {"a": 1.0, "b": np.nan, "c": 1.0})
        self.assertEqual(fractions, {"a": 1.0})


class LegacyCoaddReaderTestCase(lsst.utils.tests.TestCase):
    """Test finding inputs in a legacy ``coaddInputs`` catalog."""

    def testFindInputs(self):
        wcs = _make_wcs()
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(0, 0), lsst.geom.Extent2I(100, 100))
        valid = lsst.geom.Box2I(lsst.geom.Point2I(0, 0), lsst.geom.Extent2I(50, 100))
        ccds = _make_ccds(wcs, [(1, 10, 2.0, bbox, bbox, 57000.0),
                                (2, 11, 3.0, bbox, valid, 57001.0)])
        ra, dec = _sky(wcs, [20.0, 80.0, 150.0], [50.0, 50.0, 50.0])
        inputs = _LegacyCoaddReader(ccds).find_inputs(ra, dec)
        found = {(row["point"], row["input_visit"]) for row in inputs}
        # Point 1 is outside the valid polygon of visit 2; point 2 is
        # outside every input.
        self.assertEqual(found, {(0, 1), (0, 2), (1, 1)})
        row = inputs[(inputs["point"] == 0) & (inputs["input_visit"] == 2)][0]
        self.assertEqual(row["input_detector"], 11)
        self.assertEqual(row["coadd_weight"], 3.0)
        self.assertAlmostEqual(row["mid_mjd_tai"], 57001.0)
        self.assertEqual(row["exposure_time"], 30.0)


class CellCoaddReaderTestCase(lsst.utils.tests.TestCase):
    """Test finding inputs in a cell coadd's provenance."""

    def testFindInputs(self):
        bbox = Box.from_shape((200, 200))
        frame = TractFrame(skymap="test", tract=1, bbox=bbox)
        projection = make_random_sky_projection(np.random.default_rng(5), frame, bbox)
        grid = CellGrid(bbox=bbox, cell_shape=YX(100, 100))
        inputs = CoaddProvenance.make_empty_input_table(2)
        inputs["instrument"] = "Cam"
        inputs["physical_filter"] = "r"
        inputs["visit"] = [1, 2]
        inputs["detector"] = [5, 6]
        inputs["day_obs"] = 20250101
        inputs["polygon"] = [Polygon.from_box(bbox), Polygon.from_box(Box.from_shape((200, 50)))]
        contributions = CoaddProvenance.make_empty_contribution_table(3)
        contributions["cell_i"] = [0, 0, 1]
        contributions["cell_j"] = [0, 0, 0]
        contributions["instrument"] = "Cam"
        contributions["visit"] = [1, 2, 1]
        contributions["detector"] = [5, 6, 5]
        contributions["weight"] = [2.0, 3.0, 4.0]
        times = {("Cam", 1): (57000.0, 30.0), ("Cam", 2): (57001.0, 15.0)}
        reader = _CellCoaddReader(CoaddProvenance(inputs, contributions), projection, grid, times)
        # Point 0 is in cell (0, 0) inside both polygons; point 1 is in the
        # same cell outside the polygon of visit 2; point 2 is in cell (1, 0),
        # where only visit 1 contributes, with a weight of its own.
        sky = projection.pixel_to_sky(x=np.array([20.0, 80.0, 20.0]), y=np.array([20.0, 20.0, 150.0]))
        found = reader.find_inputs(sky.ra.deg, sky.dec.deg)
        self.assertEqual({(row["point"], row["input_visit"], row["coadd_weight"]) for row in found},
                         {(0, 1, 2.0), (0, 2, 3.0), (1, 1, 2.0), (2, 1, 4.0)})
        row = found[found["input_visit"] == 2][0]
        self.assertEqual((row["mid_mjd_tai"], row["exposure_time"]), (57001.0, 15.0))


class _FakeButler:
    """Butler that serves one template and legacy coadd inputs.

    Parameters
    ----------
    templates : `list` [`lsst.images.DifferenceImageTemplateInfo`]
        Templates of every science image.
    ccds : `dict` [`uuid.UUID`, `lsst.afw.table.ExposureCatalog`]
        Coadd ``ccds`` catalogs keyed by dataset ID.
    variances : `dict` [`uuid.UUID`, `float`]
        Constant variance of each coadd.
    wcs : `lsst.afw.geom.SkyWcs`
        WCS of every coadd.
    template_components : `set` [`str`]
        Components of the template storage class.
    """

    def __init__(self, templates, ccds, variances, wcs, template_components=("templates",)):
        self._templates = templates
        self._ccds = ccds
        self._variances = variances
        self._wcs = wcs
        self._template_components = set(template_components)

    def get_dataset_type(self, name):
        storage_class = types.SimpleNamespace(allComponents=lambda: self._template_components)
        return types.SimpleNamespace(storageClass=storage_class, storageClass_name="Test")

    def get_dataset(self, dataset_id):
        storage_class = types.SimpleNamespace(allComponents=lambda: {"coaddInputs": None})
        dataset_type = types.SimpleNamespace(storageClass=storage_class, storageClass_name="ExposureF")
        return types.SimpleNamespace(
            id=dataset_id, datasetType=dataset_type,
            makeComponentRef=lambda component: (dataset_id, component),
        )

    def get(self, ref, data_id=None, parameters=None):
        if isinstance(ref, str):
            return self._templates
        dataset_id, component = ref
        match component:
            case "coaddInputs":
                return types.SimpleNamespace(ccds=self._ccds[dataset_id])
            case "wcs":
                return self._wcs
            case "bbox":
                return lsst.geom.Box2I(lsst.geom.Point2I(0, 0), lsst.geom.Extent2I(100, 100))
            case "variance":
                image = afwImage.ImageF(parameters["bbox"])
                image.array[:, :] = self._variances[dataset_id]
                return image


class GetTemplateInputsTestCase(lsst.utils.tests.TestCase):
    """Test the full chain from diaSources to template inputs."""

    def setUp(self):
        self.wcs = _make_wcs()
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(0, 0), lsst.geom.Extent2I(100, 100))
        self.ids = [uuid.uuid4(), uuid.uuid4()]
        # Two patches of one tract cover x < 60 and x >= 40 of the science
        # image, which here shares the coadd pixel grid.
        self.templates = [
            DifferenceImageTemplateInfo(
                skymap="test", tract=1, patch=patch, dataset_id=dataset_id, dataset_run="run",
                bounds=Polygon.from_box(Box.from_shape((100, 60), start=(0, x0))),
                psf_shape_xx=1.0, psf_shape_yy=1.0, psf_shape_xy=0.0, psf_shape_flag=False,
            )
            for patch, dataset_id, x0 in [(0, self.ids[0], 0), (1, self.ids[1], 40)]
        ]
        self.ccds = {
            self.ids[0]: _make_ccds(self.wcs, [(1, 10, 1.0, bbox, bbox, 57000.0),
                                               (2, 10, 3.0, bbox, bbox, 57002.0)]),
            self.ids[1]: _make_ccds(self.wcs, [(1, 10, 1.0, bbox, bbox, 57000.0)]),
        }
        x = np.array([20.0, 50.0])
        y = np.array([50.0, 50.0])
        ra, dec = _sky(self.wcs, x, y)
        self.sources = astropy.table.Table({"diaSourceId": [100, 101], "visit": [7, 7], "detector": [3, 3],
                                            "ra": ra, "dec": dec, "x": x, "y": y})

    def testInputs(self):
        butler = _FakeButler(self.templates, self.ccds, {self.ids[0]: 1.0, self.ids[1]: 4.0}, self.wcs)
        inputs = get_template_inputs(butler, self.sources)
        # Source 100 is in patch 0 only. Source 101 is in both, which are
        # weighted 2/3 and 1/3; visit 1 reaches it through both.
        first = inputs[inputs["diaSourceId"] == 100]
        self.assertFloatsAlmostEqual(np.array(first["weight"]), np.array([0.25, 0.75]))
        second = inputs[inputs["diaSourceId"] == 101]
        self.assertEqual(list(second["patch"]), [0, 0, 1])
        self.assertFloatsAlmostEqual(np.array(second["weight"]), np.array([1/6, 1/2, 1/3]), rtol=1e-12)
        summary = summarize_template_inputs(inputs)
        self.assertEqual(list(summary["n_inputs"]), [2, 2])
        self.assertFloatsAlmostEqual(np.array(summary["mean_mjd_tai"]), np.array([57001.5, 57001.0]))

    def testLegacyTemplateRejected(self):
        butler = _FakeButler(self.templates, self.ccds, {}, self.wcs, template_components=())
        with self.assertRaisesRegex(TypeError, "legacy Exposures are not supported"):
            get_template_inputs(butler, self.sources)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

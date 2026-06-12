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

import os
import unittest
import unittest.mock as mock

import astropy.table

import lsst.afw.geom
import lsst.afw.image
import lsst.geom
import lsst.utils.tests
from lsst.analysis.ap.ppdb import (
    DEFAULT_PPDB_TAP_URL,
    DiaObjectLightCurve,
    PpdbTap,
    region_from_exposure,
)


class _FakeTapResults:
    """Stand-in for a pyvo TAPResults; just wraps a fixed astropy Table."""

    def __init__(self, table):
        self._table = table

    def to_table(self):
        return self._table


class _FakeTapService:
    """Record ADQL queries and return preset tables, without any network.

    ``tables`` may be a single Table (returned for every call) or a list of
    Tables returned for successive calls (the last is repeated thereafter).
    """

    def __init__(self, tables):
        if isinstance(tables, astropy.table.Table):
            tables = [tables]
        self._tables = list(tables)
        self.queries = []
        self.maxrecs = []

    def run_async(self, query, maxrec=None):
        self.queries.append(query)
        self.maxrecs.append(maxrec)
        index = min(len(self.queries) - 1, len(self._tables) - 1)
        return _FakeTapResults(self._tables[index])


def _objects_table(dia_object_ids=(1, 2, 3)):
    n = len(dia_object_ids)
    return astropy.table.Table({
        "diaObjectId": list(dia_object_ids),
        "ra": [150.0 + 0.1 * i for i in range(n)],
        "dec": [2.5 + 0.1 * i for i in range(n)],
    })


def _sources_table(n=2, id_column="diaSourceId"):
    return astropy.table.Table({
        id_column: [100 + i for i in range(n)],
        "diaObjectId": [42] * n,
        "midpointMjdTai": [60000.0 + i for i in range(n)],
        "band": ["r"] * n,
    })


def _make_exposure(ra=150.0, dec=2.5, scale=0.2, size=100):
    """A square ExposureF with a simple TAN WCS centered at (ra, dec)."""
    bbox = lsst.geom.Box2I(lsst.geom.Point2I(0, 0),
                           lsst.geom.Extent2I(size, size))
    exposure = lsst.afw.image.ExposureF(bbox)
    crpix = lsst.geom.Point2D(size / 2.0, size / 2.0)
    crval = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
    cd_matrix = lsst.afw.geom.makeCdMatrix(scale=scale * lsst.geom.arcseconds)
    exposure.setWcs(lsst.afw.geom.makeSkyWcs(crpix, crval, cd_matrix))
    return exposure


class TestRegionFromExposure(lsst.utils.tests.TestCase):
    def test_center_and_radius(self):
        exposure = _make_exposure(ra=150.0, dec=2.5, scale=0.2, size=100)
        ra, dec, radius0 = region_from_exposure(exposure, padding=0.0)
        self.assertAlmostEqual(ra, 150.0, places=3)
        self.assertAlmostEqual(dec, 2.5, places=3)
        # ~half-diagonal of a 100px box at 0.2 arcsec/px is ~14 arcsec.
        self.assertAlmostEqual(radius0 * 3600.0, 14.0, delta=0.5)

    def test_padding_is_arcseconds(self):
        exposure = _make_exposure()
        _, _, radius0 = region_from_exposure(exposure, padding=0.0)
        _, _, radius5 = region_from_exposure(exposure, padding=5.0)
        # padding must add exactly 5 arcsec to the radius.
        self.assertAlmostEqual((radius5 - radius0) * 3600.0, 5.0, places=6)

    def test_no_wcs_raises(self):
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(0, 0),
                               lsst.geom.Extent2I(10, 10))
        exposure = lsst.afw.image.ExposureF(bbox)
        with self.assertRaisesRegex(ValueError, "no WCS"):
            region_from_exposure(exposure)

    def test_negative_padding_raises(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            region_from_exposure(_make_exposure(), padding=-1.0)


class TestPpdbTapAuth(lsst.utils.tests.TestCase):
    def test_missing_token_raises(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "No RSP token"):
                PpdbTap()

    def test_token_sets_bearer_header(self):
        with mock.patch("lsst.analysis.ap.ppdb.pyvo.dal.TAPService") as tap:
            PpdbTap(token="secret-token")
        args, kwargs = tap.call_args
        self.assertEqual(args[0], DEFAULT_PPDB_TAP_URL)
        session = kwargs["session"]
        self.assertEqual(session.headers["Authorization"],
                         "Bearer secret-token")

    def test_env_token_used(self):
        with mock.patch.dict(os.environ, {"RSP_TOKEN": "from-env"}, clear=True):
            with mock.patch("lsst.analysis.ap.ppdb.pyvo.dal.TAPService") as tap:
                PpdbTap()
        session = tap.call_args.kwargs["session"]
        self.assertEqual(session.headers["Authorization"], "Bearer from-env")


class TestLoadObjects(lsst.utils.tests.TestCase):
    def test_default_query(self):
        fake = _FakeTapService(_objects_table())
        PpdbTap(service=fake).load_objects()
        query = fake.queries[-1]
        self.assertIn("FROM ppdb.DiaObject", query)
        self.assertIn("validityEndMjdTai IS NULL", query)
        self.assertIn("TOP 100000", query)
        self.assertIn("ORDER BY diaObjectId", query)

    def test_latest_false_drops_validity_filter(self):
        fake = _FakeTapService(_objects_table())
        PpdbTap(service=fake).load_objects(latest=False)
        self.assertNotIn("validityEndMjdTai", fake.queries[-1])

    def test_cone_search(self):
        fake = _FakeTapService(_objects_table())
        PpdbTap(service=fake).load_objects(ra=150.0, dec=2.5, radius=1.0)
        query = fake.queries[-1]
        self.assertIn("CONTAINS(POINT('ICRS', ra, dec)", query)
        self.assertIn("CIRCLE('ICRS', 150.0, 2.5, 1.0)", query)

    def test_partial_cone_raises(self):
        fake = _FakeTapService(_objects_table())
        with self.assertRaisesRegex(ValueError, "ra, dec, and radius"):
            PpdbTap(service=fake).load_objects(ra=150.0, dec=2.5)

    def test_exposure_region(self):
        fake = _FakeTapService(_objects_table())
        PpdbTap(service=fake).load_objects(exposure=_make_exposure())
        self.assertIn("CONTAINS(POINT('ICRS', ra, dec)", fake.queries[-1])

    def test_exposure_and_cone_conflict(self):
        fake = _FakeTapService(_objects_table())
        with self.assertRaisesRegex(ValueError, "either exposure="):
            PpdbTap(service=fake).load_objects(exposure=_make_exposure(),
                                               ra=150.0, dec=2.5, radius=1.0)

    def test_no_limit_omits_top(self):
        fake = _FakeTapService(_objects_table())
        PpdbTap(service=fake).load_objects(limit=None)
        self.assertNotIn("TOP", fake.queries[-1])

    def test_duplicate_current_versions_warns(self):
        fake = _FakeTapService(_objects_table((1, 1, 2)))
        ppdb = PpdbTap(service=fake)
        with self.assertLogs("lsst.analysis.ap.ppdb", level="WARNING") as cm:
            ppdb.load_objects()
        self.assertTrue(any("duplicate current versions" in m
                            for m in cm.output))

    def test_truncation_warns(self):
        fake = _FakeTapService(_objects_table((1, 2)))
        ppdb = PpdbTap(service=fake)
        with self.assertLogs("lsst.analysis.ap.ppdb", level="WARNING") as cm:
            ppdb.load_objects(limit=2)
        self.assertTrue(any("row limit" in m for m in cm.output))

    def test_load_object_missing_raises(self):
        fake = _FakeTapService(astropy.table.Table())
        with self.assertRaisesRegex(RuntimeError, "diaObjectId=999 not found"):
            PpdbTap(service=fake).load_object(999)

    def test_load_object_returns_row(self):
        fake = _FakeTapService(_objects_table((7,)))
        row = PpdbTap(service=fake).load_object(7)
        self.assertEqual(row["diaObjectId"], 7)
        self.assertIn("diaObjectId = 7", fake.queries[-1])
        self.assertIn("validityEndMjdTai IS NULL", fake.queries[-1])


class TestLoadSources(lsst.utils.tests.TestCase):
    def test_sources_for_object(self):
        fake = _FakeTapService(_sources_table())
        PpdbTap(service=fake).load_sources_for_object(42)
        query = fake.queries[-1]
        self.assertIn("FROM ppdb.DiaSource", query)
        self.assertIn("diaObjectId = 42", query)
        self.assertIn("ORDER BY midpointMjdTai", query)

    def test_forced_sources_for_object(self):
        fake = _FakeTapService(_sources_table(id_column="diaForcedSourceId"))
        PpdbTap(service=fake).load_forced_sources_for_object(42)
        self.assertIn("FROM ppdb.DiaForcedSource", fake.queries[-1])

    def test_band_filter(self):
        fake = _FakeTapService(_sources_table())
        PpdbTap(service=fake).load_sources(diaObjectId=42, bands=["r", "g"])
        query = fake.queries[-1]
        self.assertIn("diaObjectId = 42", query)
        self.assertIn("band IN ('r', 'g')", query)

    def test_invalid_band_raises(self):
        fake = _FakeTapService(_sources_table())
        with self.assertRaisesRegex(ValueError, "Unknown band"):
            PpdbTap(service=fake).load_sources(diaObjectId=42, bands=["x"])

    def test_time_window(self):
        fake = _FakeTapService(_sources_table())
        PpdbTap(service=fake).load_sources(diaObjectId=42, mjd_begin=100.0,
                                           mjd_end=200.0)
        query = fake.queries[-1]
        self.assertIn("midpointMjdTai >= 100.0", query)
        self.assertIn("midpointMjdTai < 200.0", query)

    def test_object_id_required(self):
        fake = _FakeTapService(_sources_table())
        # omitting it entirely is a TypeError (required keyword)
        with self.assertRaises(TypeError):
            PpdbTap(service=fake).load_sources()
        # passing None explicitly gives a guiding ValueError
        with self.assertRaisesRegex(ValueError, "diaObjectId is required"):
            PpdbTap(service=fake).load_sources(diaObjectId=None)

    def test_load_source_missing_raises(self):
        fake = _FakeTapService(astropy.table.Table())
        with self.assertRaisesRegex(RuntimeError, "diaSourceId=5 not found"):
            PpdbTap(service=fake).load_source(5)

    def test_load_source_returns_row(self):
        fake = _FakeTapService(_sources_table(n=1))
        row = PpdbTap(service=fake).load_source(100)
        self.assertEqual(row["diaSourceId"], 100)

    def test_id_list_is_chunked(self):
        fake = _FakeTapService(_sources_table(n=2))
        result = PpdbTap(service=fake).load_sources(
            diaObjectId=[1, 2, 3, 4, 5], id_chunk_size=2)
        # 5 ids in chunks of 2 -> 3 IN queries, results concatenated.
        self.assertEqual(len(fake.queries), 3)
        self.assertIn("diaObjectId IN (1, 2)", fake.queries[0])
        self.assertIn("diaObjectId IN (3, 4)", fake.queries[1])
        self.assertIn("diaObjectId IN (5)", fake.queries[2])
        self.assertEqual(len(result), 6)

    def test_empty_id_list_returns_empty(self):
        fake = _FakeTapService(_sources_table(n=0))
        result = PpdbTap(service=fake).load_sources(diaObjectId=[])
        self.assertEqual(len(result), 0)
        self.assertIn("1 = 0", fake.queries[-1])


class TestRegionAndCone(lsst.utils.tests.TestCase):
    def test_sources_for_region_two_step(self):
        fake = _FakeTapService([_objects_table((10, 20, 30)),
                                _sources_table(n=2)])
        PpdbTap(service=fake).load_sources_for_region(ra=150.0, dec=2.5,
                                                      radius=0.1)
        # first: cone on DiaObject (the only table allowing spatial search).
        obj_query = fake.queries[0]
        self.assertIn("FROM ppdb.DiaObject", obj_query)
        self.assertIn("validityEndMjdTai IS NULL", obj_query)
        self.assertIn("CONTAINS(POINT('ICRS', ra, dec)", obj_query)
        # second: sources by id, with no spatial predicate.
        src_query = fake.queries[1]
        self.assertIn("FROM ppdb.DiaSource", src_query)
        self.assertIn("diaObjectId IN (10, 20, 30)", src_query)
        self.assertNotIn("CONTAINS", src_query)

    def test_forced_sources_for_region_with_exposure(self):
        fake = _FakeTapService([
            _objects_table((7,)),
            _sources_table(n=1, id_column="diaForcedSourceId"),
        ])
        PpdbTap(service=fake).load_forced_sources_for_region(
            exposure=_make_exposure())
        self.assertIn("FROM ppdb.DiaObject", fake.queries[0])
        self.assertIn("CONTAINS", fake.queries[0])
        self.assertIn("FROM ppdb.DiaForcedSource", fake.queries[1])
        self.assertIn("diaObjectId IN (7)", fake.queries[1])

    def test_region_with_no_objects(self):
        fake = _FakeTapService([astropy.table.Table(), _sources_table(n=0)])
        result = PpdbTap(service=fake).load_sources_for_region(
            ra=1.0, dec=1.0, radius=0.01)
        self.assertEqual(len(result), 0)
        self.assertIn("1 = 0", fake.queries[1])

    def test_sources_by_cone_warns_and_queries_source_table(self):
        fake = _FakeTapService(_sources_table(n=2))
        ppdb = PpdbTap(service=fake)
        with self.assertLogs("lsst.analysis.ap.ppdb", level="WARNING") as cm:
            ppdb.load_sources_by_cone(ra=150.0, dec=2.5, radius=0.05)
        self.assertTrue(any("prototype-only" in m for m in cm.output))
        query = fake.queries[-1]
        self.assertIn("FROM ppdb.DiaSource", query)
        self.assertIn("CONTAINS(POINT('ICRS', ra, dec)", query)

    def test_forced_sources_by_cone(self):
        fake = _FakeTapService(
            _sources_table(n=1, id_column="diaForcedSourceId"))
        ppdb = PpdbTap(service=fake)
        with self.assertLogs("lsst.analysis.ap.ppdb", level="WARNING"):
            ppdb.load_forced_sources_by_cone(exposure=_make_exposure())
        self.assertIn("FROM ppdb.DiaForcedSource", fake.queries[-1])
        self.assertIn("CONTAINS", fake.queries[-1])

    def test_by_cone_requires_a_cone(self):
        fake = _FakeTapService(_sources_table())
        with self.assertRaisesRegex(ValueError, "cone search requires"):
            PpdbTap(service=fake).load_sources_by_cone()

    def test_sources_by_time_window_warns_and_queries_source_table(self):
        fake = _FakeTapService(_sources_table(n=2))
        ppdb = PpdbTap(service=fake)
        with self.assertLogs("lsst.analysis.ap.ppdb", level="WARNING") as cm:
            ppdb.load_sources_by_time_window(mjd_begin=60000.0, mjd_end=60001.0)
        self.assertTrue(any("prototype-only" in m for m in cm.output))
        query = fake.queries[-1]
        self.assertIn("FROM ppdb.DiaSource", query)
        self.assertIn("midpointMjdTai >= 60000.0", query)
        self.assertIn("midpointMjdTai < 60001.0", query)
        self.assertNotIn("diaObjectId", query)
        self.assertNotIn("CONTAINS", query)

    def test_forced_sources_by_time_window_open_ended(self):
        fake = _FakeTapService(
            _sources_table(n=1, id_column="diaForcedSourceId"))
        ppdb = PpdbTap(service=fake)
        with self.assertLogs("lsst.analysis.ap.ppdb", level="WARNING"):
            ppdb.load_forced_sources_by_time_window(mjd_begin=60000.0)
        query = fake.queries[-1]
        self.assertIn("FROM ppdb.DiaForcedSource", query)
        self.assertIn("midpointMjdTai >= 60000.0", query)

    def test_by_time_window_requires_a_bound(self):
        fake = _FakeTapService(_sources_table())
        with self.assertRaisesRegex(ValueError, "time-window search requires"):
            PpdbTap(service=fake).load_sources_by_time_window()


class TestColumnOrdering(lsst.utils.tests.TestCase):
    def test_sources_columns_reordered_to_sdm(self):
        # alphabetical input columns -> SDM-schema order out.
        scrambled = astropy.table.Table({
            "band": ["r"], "dec": [1.0], "diaObjectId": [42],
            "diaSourceId": [100], "midpointMjdTai": [60000.0], "ra": [150.0]})
        fake = _FakeTapService(scrambled)
        result = PpdbTap(service=fake).load_sources(diaObjectId=42)
        self.assertEqual(result.colnames,
                         ["diaSourceId", "diaObjectId", "midpointMjdTai",
                          "ra", "dec", "band"])

    def test_unknown_columns_raise(self):
        # a column not in the SDM schema signals schema drift -> error.
        tab = astropy.table.Table({
            "ra": [150.0], "diaSourceId": [100], "myExtra": [1]})
        fake = _FakeTapService(tab)
        with self.assertRaisesRegex(RuntimeError, "not in the SDM schema"):
            PpdbTap(service=fake).load_sources(diaObjectId=42)

    def test_explicit_columns_order_preserved(self):
        # an explicit columns list is respected, not reordered to SDM order.
        tab = astropy.table.Table({"dec": [1.0], "ra": [150.0]})
        fake = _FakeTapService(tab)
        result = PpdbTap(service=fake).load_sources(diaObjectId=42,
                                                    columns=["dec", "ra"])
        self.assertEqual(result.colnames, ["dec", "ra"])

    def test_objects_columns_reordered_to_sdm(self):
        scrambled = astropy.table.Table({
            "ra": [150.0], "diaObjectId": [1], "dec": [2.0],
            "validityEndMjdTai": [60000.0]})
        fake = _FakeTapService(scrambled)
        result = PpdbTap(service=fake).load_objects()
        self.assertEqual(
            result.colnames,
            ["diaObjectId", "validityEndMjdTai", "ra", "dec"])

    def test_load_object_row_in_sdm_order(self):
        scrambled = astropy.table.Table({
            "ra": [150.0], "diaObjectId": [7], "dec": [2.0]})
        fake = _FakeTapService(scrambled)
        row = PpdbTap(service=fake).load_object(7)
        self.assertEqual(row.colnames, ["diaObjectId", "ra", "dec"])

    def test_id_list_path_reordered_to_sdm(self):
        # the chunked many-ids path reorders per page before stacking.
        scrambled = astropy.table.Table({
            "band": ["r"], "diaObjectId": [42], "diaSourceId": [100],
            "midpointMjdTai": [60000.0]})
        fake = _FakeTapService(scrambled)
        result = PpdbTap(service=fake).load_sources(diaObjectId=[1, 2],
                                                    id_chunk_size=1)
        self.assertEqual(
            result.colnames,
            ["diaSourceId", "diaObjectId", "midpointMjdTai", "band"])


class TestLightCurve(lsst.utils.tests.TestCase):
    def test_assembles_object_and_series(self):
        tables = [
            _objects_table((42,)),
            _sources_table(n=2),
            _sources_table(n=3, id_column="diaForcedSourceId"),
        ]
        fake = _FakeTapService(tables)
        light_curve = PpdbTap(service=fake).load_light_curve(42)
        self.assertIsInstance(light_curve, DiaObjectLightCurve)
        self.assertEqual(light_curve.diaObjectId, 42)
        self.assertEqual(light_curve.n_sources, 2)
        self.assertEqual(light_curve.n_forced_sources, 3)
        # one query each for object, sources, forced sources.
        self.assertEqual(len(fake.queries), 3)
        self.assertIn("FROM ppdb.DiaObject", fake.queries[0])


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

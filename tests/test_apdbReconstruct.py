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

import unittest

import lsst.utils.tests
import numpy as np
import pandas as pd

from lsst.analysis.ap.apdbReconstruct import (
    ApdbReconstructor,
    InMemoryDbQuery,
)


def _diaSources():
    """Two distinct diaSourceIds, plus one duplicate (later visit wins)."""
    return pd.DataFrame({
        "diaSourceId": [100, 200, 100],
        "diaObjectId": [10, 20, 10],
        "visit": [1, 2, 3],
        "detector": [50, 50, 50],
        "ra": [1.0, 2.0, 1.5],
        "dec": [-1.0, -2.0, -1.5],
        "midpointMjdTai": [60100.0, 60101.0, 60110.0],
        "psfFlux": [100.0, 200.0, 150.0],
        "psfFluxErr": [10.0, 20.0, 15.0],
        "band": ["g", "r", "g"],
        "x": [1.0, 2.0, 1.0],
        "y": [1.0, 2.0, 1.0],
        "psfNdata": [10, 10, 10],
    })


def _diaObjects():
    """diaObjectId=10 appears twice with different validityStart; the later
    snapshot should win after dedup. diaObjectId=20 appears once.
    """
    return pd.DataFrame({
        "diaObjectId": [10, 20, 10],
        "validityStartMjdTai": [60100.0, 60101.0, 60110.0],
        "ra": [1.0, 2.0, 1.5],
        "dec": [-1.0, -2.0, -1.5],
        "nDiaSources": [1, 1, 2],
    })


def _diaForcedSources():
    """Three rows with the second duplicating the first on PK (latest wins)."""
    return pd.DataFrame({
        "diaForcedSourceId": [1000, 1001, 1002],
        "diaObjectId": [10, 10, 20],
        "visit": [1, 1, 2],          # row 0 and row 1 share PK
        "detector": [50, 50, 50],
        "midpointMjdTai": [60100.0, 60100.0, 60101.0],
        "psfFlux": [100.0, 150.0, 200.0],  # later (150) should win
        "psfFluxErr": [10.0, 15.0, 20.0],
        "band": ["g", "g", "r"],
        "ra": [1.0, 1.0, 2.0],
        "dec": [-1.0, -1.0, -2.0],
        "scienceFlux": [50.0, 75.0, 100.0],
        "scienceFluxErr": [5.0, 7.0, 10.0],
        "timeProcessedMjdTai": [60101.0, 60102.0, 60102.0],
    })


class TestFinalize(lsst.utils.tests.TestCase):
    """Tests for the staticmethod that does dedup + schema coercion."""

    def test_diaSource_dedup_by_id(self):
        result = ApdbReconstructor.finalize(
            _diaSources(), _diaObjects(), _diaForcedSources(),
            coerce_to_schema=False)
        # 3 input rows -> 2 unique diaSourceIds.
        self.assertEqual(len(result.diaSources), 2)
        self.assertEqual(sorted(result.diaSources["diaSourceId"]), [100, 200])
        # The duplicate-keep="last" entry should win: visit 3 wins over visit 1
        row100 = result.diaSources.set_index("diaSourceId").loc[100]
        self.assertEqual(int(row100["visit"]), 3)

    def test_diaForcedSource_dedup_by_pk(self):
        result = ApdbReconstructor.finalize(
            _diaSources(), _diaObjects(), _diaForcedSources(),
            coerce_to_schema=False)
        # 3 input rows -> 2 unique (diaObjectId, visit, detector).
        self.assertEqual(len(result.diaForcedSources), 2)
        # The later row for the dup PK should win (psfFlux=150, not 100).
        dup = result.diaForcedSources[
            (result.diaForcedSources["diaObjectId"] == 10)
            & (result.diaForcedSources["visit"] == 1)
            & (result.diaForcedSources["detector"] == 50)]
        self.assertEqual(len(dup), 1)
        self.assertAlmostEqual(float(dup["psfFlux"].iloc[0]), 150.0)

    def test_diaObject_keeps_latest_by_validity(self):
        result = ApdbReconstructor.finalize(
            _diaSources(), _diaObjects(), _diaForcedSources(),
            coerce_to_schema=False)
        self.assertEqual(len(result.diaObjects), 2)
        # diaObject 10 had two snapshots; the later (validityStart=60110)
        # should win — nDiaSources=2, not 1.
        row10 = result.diaObjects.set_index("diaObjectId").loc[10]
        self.assertEqual(int(row10["nDiaSources"]), 2)

    def test_diaObject_dedup_skips_nan_nDiaSources(self):
        """Passthrough snapshots from quanta that touch a diaObject but
        don't actually update it leave ``nDiaSources`` as NaN. The dedup
        must skip those in favor of an older snapshot that carries the
        real count — otherwise the survivor's NaN gets fillna(0)'d during
        schema coercion and the user sees ``nDiaSources=0`` even though
        the diaObject has real diaSources.
        """
        diaObjects = pd.DataFrame({
            "diaObjectId":         [10, 10, 10, 20],  # noqa: E241
            "validityStartMjdTai": [0.0, 61167.0, 61168.0, 61167.0],  # noqa: E241
            "nDiaSources":         [1.0, np.nan, np.nan, 2.0],  # noqa: E241
            "ra":                  [1.0, 1.0, 1.0, 2.0],  # noqa: E241
            "dec":                 [-1.0, -1.0, -1.0, -2.0],  # noqa: E241
        })
        empty = pd.DataFrame()
        result = ApdbReconstructor.finalize(
            empty, diaObjects, empty, coerce_to_schema=False)
        # diaObject 10: must survive with nDiaSources=1 (NOT NaN, NOT 0).
        row10 = result.diaObjects.set_index("diaObjectId").loc[10]
        self.assertEqual(int(row10["nDiaSources"]), 1)
        # diaObject 20: untouched (only one row), should round-trip.
        row20 = result.diaObjects.set_index("diaObjectId").loc[20]
        self.assertEqual(int(row20["nDiaSources"]), 2)

    def test_diaObject_dedup_prefers_higher_nDiaSources(self):
        """When a diaObject has multiple informative snapshots, dedup
        picks the one with the most diaSources (which is also typically
        the latest update; this ordering is robust against snapshot
        re-ordering during concatenation).
        """
        diaObjects = pd.DataFrame({
            "diaObjectId":         [10, 10, 10],  # noqa: E241
            "validityStartMjdTai": [60100.0, 60110.0, 60120.0],  # noqa: E241
            "nDiaSources":         [1.0, 3.0, 2.0],    # noqa: E241
            "ra":                  [1.0, 1.0, 1.0],  # noqa: E241
            "dec":                 [-1.0, -1.0, -1.0],  # noqa: E241
        })
        empty = pd.DataFrame()
        result = ApdbReconstructor.finalize(
            empty, diaObjects, empty, coerce_to_schema=False)
        row10 = result.diaObjects.set_index("diaObjectId").loc[10]
        # The snapshot with nDiaSources=3 wins (highest count), even
        # though a later validity is available.
        self.assertEqual(int(row10["nDiaSources"]), 3)

    def test_diaObject_history_keeps_all(self):
        result = ApdbReconstructor.finalize(
            _diaSources(), _diaObjects(), _diaForcedSources(),
            coerce_to_schema=False, history=True)
        # Full update trail preserved.
        self.assertEqual(len(result.diaObjects), 3)

    def test_schema_coercion_dtypes(self):
        result = ApdbReconstructor.finalize(
            _diaSources(), _diaObjects(), _diaForcedSources(),
            coerce_to_schema=True)
        # Integer IDs come out as Int64 (nullable long) or int64.
        # diaSourceId is non-nullable -> int64;
        # diaObjectId is nullable -> Int64.
        self.assertEqual(str(result.diaSources["diaSourceId"].dtype), "int64")
        self.assertEqual(str(result.diaSources["diaObjectId"].dtype), "Int64")
        # Coercion also fills in schema columns that were missing — these
        # ought to appear with sensible defaults.
        self.assertIn("snr", result.diaSources.columns)
        # Extra columns NOT in the schema are dropped by
        # convertDataFrameToSdmSchema.
        # (We didn't introduce any in the fixtures, but verify the call
        # didn't add a bogus index column.)
        self.assertNotIn("index", result.diaSources.columns)

    def test_empty_inputs(self):
        empty = pd.DataFrame()
        result = ApdbReconstructor.finalize(empty, empty, empty,
                                            coerce_to_schema=False)
        self.assertEqual(len(result.diaSources), 0)
        self.assertEqual(len(result.diaObjects), 0)
        self.assertEqual(len(result.diaForcedSources), 0)


class TestInMemoryDbQuery(lsst.utils.tests.TestCase):
    """Tests that the DbQuery adapter routes queries against the underlying
    DataFrames the way the SQL backends do.
    """

    def setUp(self):
        recon = ApdbReconstructor.finalize(
            _diaSources(), _diaObjects(), _diaForcedSources(),
            coerce_to_schema=False)
        self.query = InMemoryDbQuery(recon.diaSources,
                                     recon.diaObjects,
                                     recon.diaForcedSources)

    def test_load_sources_for_object(self):
        result = self.query.load_sources_for_object(10)
        self.assertEqual(len(result), 1)
        self.assertEqual(int(result["diaSourceId"].iloc[0]), 100)

    def test_load_forced_sources_for_object_ignores_exclude_flagged(self):
        # DiaForcedSource has no flag columns; exclude_flagged is a no-op
        # for parity with the abstract interface.
        result = self.query.load_forced_sources_for_object(
            10, exclude_flagged=True)
        self.assertEqual(len(result), 1)
        self.assertEqual(int(result["visit"].iloc[0]), 1)

    def test_load_source_raises_when_missing(self):
        with self.assertRaisesRegex(RuntimeError, "diaSourceId=999999"):
            self.query.load_source(999999)

    def test_load_object_round_trip(self):
        obj = self.query.load_object(10)
        self.assertEqual(int(obj["diaObjectId"]), 10)
        self.assertEqual(int(obj["nDiaSources"]), 2)

    def test_load_forced_source_round_trip(self):
        result = self.query.load_forced_source(1002)
        self.assertEqual(int(result["diaForcedSourceId"]), 1002)
        self.assertEqual(int(result["visit"]), 2)

    def test_excluded_flag_validation(self):
        with self.assertRaisesRegex(ValueError, "not present"):
            self.query.set_excluded_diaSource_flags(["pixelFlags_bad"])

    def test_load_sources_with_exclude_flagged(self):
        # Add a flag column to one row and verify it's excluded.
        diaSrc = _diaSources()
        diaSrc["pixelFlags_bad"] = [False, True, False]
        recon = ApdbReconstructor.finalize(
            diaSrc, _diaObjects(), _diaForcedSources(),
            coerce_to_schema=False)
        q = InMemoryDbQuery(recon.diaSources, recon.diaObjects,
                            recon.diaForcedSources)
        q.set_excluded_diaSource_flags(["pixelFlags_bad"])
        # No exclusion requested: all 2 deduped rows returned.
        self.assertEqual(len(q.load_sources()), 2)
        # Exclusion requested: the flagged row is dropped.
        flagged = q.load_sources(exclude_flagged=True)
        self.assertEqual(len(flagged), 1)
        self.assertNotIn(200, flagged["diaSourceId"].tolist())


class TestDatasetNameDefaults(lsst.utils.tests.TestCase):
    """Pin the dataset-name defaults to the ApPipe.yaml `associateApdb`
    config, since downstream production tooling relies on these names.
    """

    def test_default_names_match_ap_pipe(self):
        # The "apdb" entries come from ApPipe.yaml's `associateApdb` task
        # config; the "preloaded_*" entries come from
        # `lsst.ap.association.LoadDiaCatalogsTask` output names. Both
        # are loaded so the reconstruction includes both prior history
        # and the current run's new rows.
        recon = ApdbReconstructor(butler=None)
        self.assertEqual(recon.dataset_names["diaSource"],
                         ["dia_source_apdb", "preloaded_dia_source"])
        self.assertEqual(recon.dataset_names["diaObject"],
                         ["dia_object_apdb", "preloaded_dia_object"])
        self.assertEqual(recon.dataset_names["diaForcedSource"],
                         ["dia_forced_source_apdb",
                          "preloaded_dia_forced_source"])

    def test_dataset_names_override(self):
        # A full dict replaces DEFAULT_DATASET_NAMES wholesale.
        override = {
            "diaSource": ["goodSeeingDiff_assocDiaSrc"],
            "diaObject": ["goodSeeingDiff_diaObject"],
            "diaForcedSource": ["goodSeeingDiff_diaForcedSrc"],
        }
        recon = ApdbReconstructor(butler=None, dataset_names=override)
        self.assertEqual(recon.dataset_names, override)

    def test_dataset_names_override_list(self):
        # A list override replaces the default list entirely.
        recon = ApdbReconstructor(
            butler=None,
            dataset_names={"diaSource": ["a", "b", "c"]})
        self.assertEqual(recon.dataset_names["diaSource"], ["a", "b", "c"])


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

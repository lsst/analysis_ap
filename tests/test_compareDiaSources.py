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
import tempfile
import unittest
from unittest import mock

import astropy.table
import numpy as np

import lsst.afw.table as afwTable
import lsst.daf.butler.tests as butlerTests
import lsst.geom
import lsst.utils.tests
from lsst.analysis.ap.compareDiaSources import compare_dia_sources, plot_dia_source_comparison

RA0, DEC0 = 10.0, -5.0
ARCSEC = 1/3600


def _position(slot, offset=0.0):
    """Sky position of a test source: slots are 10 arcsec apart in RA, and
    ``offset`` moves the source north, in arcseconds.
    """
    return RA0 + 10*slot*ARCSEC/np.cos(np.radians(DEC0)), DEC0 + offset*ARCSEC


def _table(rows):
    """Make a diaSource table from (id, visit, detector, slot, offset,
    isNegative, psfFlux, pixelFlags_bad) rows.
    """
    columns = list(zip(*rows))
    ra, dec = zip(*(_position(slot, offset) for slot, offset in zip(columns[3], columns[4])))
    n = len(rows)
    return astropy.table.Table({
        "diaSourceId": np.array(columns[0], dtype=np.int64),
        "visit": np.array(columns[1], dtype=np.int64),
        "detector": np.array(columns[2], dtype=np.int64),
        "ra": np.array(ra),
        "dec": np.array(dec),
        "x": np.full(n, 50.0) + 5*np.array(columns[4]),
        "y": np.full(n, 60.0),
        "isNegative": np.array(columns[5], dtype=bool),
        "psfFlux": np.array(columns[6], dtype=float),
        "psfFluxErr": np.full(n, 10.0),
        "snr": np.array(columns[6], dtype=float)/10,
        "reliability": np.full(n, 0.9),
        "pixelFlags_bad": np.array(columns[7], dtype=bool),
        "band": np.full(n, "g"),
    })


def _catalog(sources):
    """Make a catalog of detections from (id, slot, reliability) entries."""
    schema = afwTable.SourceTable.makeMinimalSchema()
    negative = schema.addField("is_negative", type="Flag", doc="Negative detection.")
    reliability = schema.addField("reliability", type="F", doc="Reliability score.")
    catalog = afwTable.SourceCatalog(schema)
    for sourceId, slot, score in sources:
        record = catalog.addNew()
        record.setId(sourceId)
        record.setCoord(lsst.geom.SpherePoint(*_position(slot), lsst.geom.degrees))
        record.set(negative, False)
        record.set(reliability, score)
    return catalog


def _make_repo(path):
    butler = butlerTests.makeTestRepo(path, {"instrument": ["notACam"], "visit": [1, 2],
                                             "detector": [1, 2]})
    butler.registry.registerRun("run")
    butlerTests.addDatasetType(butler, "dia_source_visit", {"instrument", "visit"}, "ArrowAstropy")
    butlerTests.addDatasetType(butler, "dia_source_detector", {"instrument", "visit", "detector"},
                               "ArrowAstropy")
    for name in ("dia_source_low_reliability", "rejected_dia_source", "dia_source_unfiltered"):
        butlerTests.addDatasetType(butler, name, {"instrument", "visit", "detector"}, "SourceCatalog")
    butlerTests.addDatasetType(butler, "long_trailed_source_detector", {"instrument", "visit", "detector"},
                               "ArrowAstropy")
    return butler.clone(collections="run", run="run")


def _write_runs(root):
    """Write the reference and new runs and return their butlers.

    Visit 1, matched pairs: 101-201 (0.05 arcsec apart), 102-202 (on
    different detectors, with a flux and flag change), 103-203 (of opposite
    signs) and 104-204. 105 is a second detection of the 104/204 source, so
    it is unmatched. 205-209 are new only, and the reference run's
    catalogs explain why. Visit 2: the new run wrote only detector tables,
    for detector 1, so 112 on detector 2 has no data in it.
    """
    ref = _make_repo(os.path.join(root, "ref"))
    new = _make_repo(os.path.join(root, "new"))
    put = dict(instrument="notACam")
    ref.put(_table([(101, 1, 1, 0, 0.0, False, 100.0, False),
                    (102, 1, 1, 1, 0.0, False, 200.0, False),
                    (103, 1, 2, 2, 0.0, True, -50.0, False),
                    (104, 1, 1, 3, 0.0, False, 300.0, False),
                    (105, 1, 1, 3, 0.3, False, 30.0, False)]),
            "dia_source_visit", visit=1, **put)
    ref.put(_table([(111, 2, 1, 10, 0.0, False, 100.0, False),
                    (112, 2, 2, 11, 0.0, False, 100.0, False)]),
            "dia_source_visit", visit=2, **put)
    ref.put(_catalog([(901, 4, 0.2)]), "dia_source_low_reliability", visit=1, detector=1, **put)
    ref.put(_catalog([(902, 5, np.nan)]), "rejected_dia_source", visit=1, detector=1, **put)
    ra, dec = np.radians(_position(8))
    ref.put(astropy.table.Table({"id": [903], "coord_ra": [ra], "coord_dec": [dec], "is_negative": [False]}),
            "long_trailed_source_detector", visit=1, detector=1, **put)
    # Removed detections keep their IDs from the detection catalog.
    ref.put(_catalog([(910, 0, np.nan), (911, 1, np.nan), (913, 3, np.nan), (901, 4, np.nan),
                      (902, 5, np.nan), (906, 6, np.nan), (903, 8, np.nan)]),
            "dia_source_unfiltered", visit=1, detector=1, **put)

    new.put(_table([(201, 1, 1, 0, 0.05, False, 100.0, False),
                    (202, 1, 2, 1, 0.0, False, 230.0, True),
                    (203, 1, 2, 2, 0.0, False, 50.0, False),
                    (204, 1, 1, 3, 0.1, False, 300.0, False),
                    (205, 1, 1, 4, 0.0, False, 60.0, False),
                    (206, 1, 1, 5, 0.0, False, 60.0, False),
                    (207, 1, 1, 6, 0.0, False, 60.0, False),
                    (208, 1, 1, 7, 0.0, False, 60.0, False),
                    (209, 1, 1, 8, 0.0, False, 60.0, False)]),
            "dia_source_visit", visit=1, **put)
    new.put(_table([(211, 2, 1, 10, 0.0, False, 100.0, False)]),
            "dia_source_detector", visit=2, detector=1, **put)
    return ref, new


class CompareDiaSourcesTestCase(lsst.utils.tests.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.TemporaryDirectory()
        cls.ref, cls.new = _write_runs(cls.root.name)
        cls.result = compare_dia_sources(cls.ref, cls.new)

    @classmethod
    def tearDownClass(cls):
        cls.root.cleanup()

    def testSummary(self):
        summary = self.result.summary.set_index("visit")
        self.assertEqual(list(summary.index), [1, 2])
        self.assertEqual(summary.loc[1, ["n_ref", "n_new", "n_matched", "n_ref_only", "n_new_only"]].tolist(),
                         [5, 9, 4, 1, 5])
        self.assertEqual(summary.loc[2, ["n_ref", "n_new", "n_matched", "n_ref_only", "n_new_only"]].tolist(),
                         [2, 1, 1, 1, 0])
        self.assertEqual(summary.loc[1, "table_new"], "dia_source_visit")
        self.assertEqual(summary.loc[2, "table_new"], "dia_source_detector")
        for visit in (1, 2):
            expected = self.ref.registry.expandDataId(instrument="notACam", visit=visit)["band"]
            self.assertEqual(summary.loc[visit, "band"], expected)

    def testMatched(self):
        matched = self.result.matched.set_index("diaSourceId_ref")
        self.assertEqual(dict(matched["diaSourceId_new"]), {101: 201, 102: 202, 103: 203, 104: 204, 111: 211})
        self.assertAlmostEqual(matched.loc[101, "separation_arcsec"], 0.05, places=6)
        self.assertAlmostEqual(matched.loc[101, "dx"], 0.25)
        self.assertTrue(np.isnan(matched.loc[102, "dx"]))
        self.assertAlmostEqual(matched.loc[102, "delta_psfFlux_sigma"], 30/np.sqrt(200))
        self.assertEqual(matched.loc[102, "flags_changed"], "pixelFlags_bad")
        self.assertEqual(matched.loc[101, "flags_changed"], "")
        self.assertEqual(matched.loc[103, "flags_changed"], "isNegative")

    def testReasons(self):
        ref_only = self.result.ref_only.set_index("diaSourceId")
        self.assertEqual(dict(ref_only["reason"]),
                         {105: "matched_elsewhere", 112: "no_data"})
        self.assertEqual(ref_only.loc[105, "other_diaSourceId"], 204)
        new_only = self.result.new_only.set_index("diaSourceId")
        self.assertEqual(dict(new_only["reason"]),
                         {205: "low_reliability", 206: "rejected",
                          207: "removed_later", 208: "not_detected", 209: "long_trail"})
        self.assertEqual(new_only.loc[209, "other_diaSourceId"], 903)
        self.assertEqual(new_only.loc[205, "other_diaSourceId"], 901)
        self.assertAlmostEqual(new_only.loc[205, "other_reliability"], 0.2, places=6)
        self.assertEqual(set(new_only["instrument"]), {"notACam"})

    def testFlagChanges(self):
        self.assertEqual(self.result.flag_changes.to_dict("records"),
                         [{"flag": "isNegative", "n_ref_only": 1, "n_new_only": 0},
                          {"flag": "pixelFlags_bad", "n_ref_only": 0, "n_new_only": 1}])

    def testReasonCounts(self):
        counts = self.result.reason_counts()
        self.assertEqual(list(counts.index), ["matched_elsewhere", "low_reliability", "rejected",
                                              "long_trail", "removed_later", "not_detected", "no_data"])
        self.assertEqual(counts.loc["matched_elsewhere"].tolist(), [1, 0])

    def testOptions(self):
        result = compare_dia_sources(self.ref, self.new, visits=[2], explain_unmatched=False)
        self.assertEqual(list(result.summary["visit"]), [2])
        self.assertEqual(list(result.ref_only["reason"]), [None])

    def testQueries(self):
        """Catalogs are found with one query per dataset type, and detector
        tables are only queried for visits without a visit table.
        """
        with mock.patch.object(self.ref, "query_datasets", wraps=self.ref.query_datasets) as query_ref, \
                mock.patch.object(self.new, "query_datasets", wraps=self.new.query_datasets) as query_new, \
                mock.patch.object(self.ref, "find_dataset", side_effect=AssertionError), \
                mock.patch.object(self.new, "find_dataset", side_effect=AssertionError):
            compare_dia_sources(self.ref, self.new, visits=[1, 2])
        queried_ref = [call.args[0] for call in query_ref.call_args_list]
        queried_new = [call.args[0] for call in query_new.call_args_list]
        # Both runs have a visit table for visit 1, but only the reference
        # run has one for visit 2.
        self.assertNotIn("dia_source_detector", queried_ref)
        self.assertEqual(queried_new.count("dia_source_detector"), 1)
        self.assertEqual(query_new.call_args_list[queried_new.index("dia_source_detector")].kwargs["bind"],
                         {"visits": [2]})
        for dataset in ("dia_source_low_reliability", "rejected_dia_source", "long_trailed_source_detector",
                        "dia_source_unfiltered"):
            self.assertEqual(queried_ref.count(dataset), 1)
            self.assertEqual(queried_new.count(dataset), 1)

    def testPlot(self):
        figure = plot_dia_source_comparison(self.result, labels=("main", "ticket"))
        self.assertEqual(len(figure.axes), 4)

    def testNoSources(self):
        with tempfile.TemporaryDirectory() as root:
            empty = _make_repo(os.path.join(root, "empty"))
            with self.assertRaises(LookupError):
                compare_dia_sources(empty, empty)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

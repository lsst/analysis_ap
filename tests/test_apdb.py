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
import tempfile

import pandas as pd

import lsst.utils.tests
from lsst.analysis.ap.apdb import ApdbSqliteQuery

from lsst.obs.lsst import LsstCamImSim
import lsst.daf.butler


class TestApdbSqlite(lsst.utils.tests.TestCase):
    def setUp(self):
        datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/")
        apdb_file = os.path.join(datadir, "apdb.sqlite3")

        # TODO DM-39501: necessary until we can get detector/visit from APDB.
        self.path = tempfile.TemporaryDirectory()
        lsst.daf.butler.Butler.makeRepo(self.path.name)
        butler = lsst.daf.butler.Butler(self.path.name, writeable=True)
        LsstCamImSim().register(butler.registry, update=True)

        self.apdb = ApdbSqliteQuery(apdb_file, butler=butler, instrument="LSSTCam-imSim")

    def tearDown(self):
        self.path.cleanup()

    def test_load_sources(self):
        result = self.apdb.load_sources(limit=None)
        self.assertEqual(len(result), 499)
        # spot check a few fields
        self.assertEqual(result['diaSourceId'][0], 506428274000265706)
        self.assertTrue(all(result['ccdVisitId'][:297] == 943296168))
        self.assertTrue(all(result['ccdVisitId'][297:] == 982985164))
        self.assertEqual(result['diaObjectId'][14], 506428274000265720)
        self.assertEqual(result['detector'][0], 168)
        self.assertEqual(result['visit'][0], 943296)

        # check using a query limit
        result = self.apdb.load_sources(limit=2)
        self.assertEqual(len(result), 2)

    def test_load_sources_exclude_flags(self):
        result = self.apdb.load_sources(exclude_flagged=True)
        self.assertEqual(len(result), 415)

    def test_load_sources_for_object(self):
        result = self.apdb.load_sources_for_object(506428274000265761)
        # This test APDB has up to two sources per object.
        self.assertEqual(len(result), 2)
        self.assertEqual(result['diaSourceId'][0], 506428274000265761)

    def test_load_sources_for_object_exclude_flags(self):
        # diaObjectId chosen from inspection to have flagged diaSources
        result = self.apdb.load_sources_for_object(506428274000265784)
        self.assertEqual(len(result), 2)
        self.assertEqual(result['diaSourceId'][0], 506428274000265784)

        result = self.apdb.load_sources_for_object(506428274000265784,
                                                   exclude_flagged=True)
        self.assertEqual(len(result), 0)

    def test_load_objects(self):
        result = self.apdb.load_objects(limit=None)
        self.assertEqual(len(result), 436)
        # spot check a few fields
        self.assertNotIn("diaSourceId", result)
        self.assertEqual(result['diaObjectId'][0], 506428274000265706)
        self.assertIn("validityStart", result.columns)

        result = self.apdb.load_objects(limit=2)
        self.assertEqual(len(result), 2)

    def test_load_forced_sources(self):
        result = self.apdb.load_forced_sources(limit=None)
        self.assertEqual(len(result), 653)
        # spot check a few fields
        self.assertEqual(result['diaObjectId'][0], 506428274000265706)
        self.assertEqual(result['diaForcedSourceId'][0], 506428274000265217)
        self.assertEqual(result['detector'][0], 168)
        self.assertEqual(result['visit'][0], 943296)

        result = self.apdb.load_forced_sources(limit=2)
        self.assertEqual(len(result), 2)

    def test_load_source(self):
        result = self.apdb.load_source(506428274000265709)
        # spot check a few fields
        self.assertEqual(result['diaSourceId'], 506428274000265709)
        self.assertEqual(result['diaObjectId'], 506428274000265709)
        self.assertEqual(result['flags'], 8388608)

        with self.assertRaisesRegex(RuntimeError, "diaSourceId=54321 not found"):
            self.apdb.load_source(54321)

    def test_load_object(self):
        result = self.apdb.load_object(506428274000265714)
        # spot check a few fields
        self.assertEqual(result['diaObjectId'], 506428274000265714)
        self.assertFloatsAlmostEqual(result['ra'], 55.7302576718241, rtol=1e-15)

        with self.assertRaisesRegex(RuntimeError, "diaObjectId=54321 not found"):
            self.apdb.load_object(54321)

    def test_load_forced_source(self):
        result = self.apdb.load_forced_source(506428274000265224)
        # spot check a few fields
        self.assertEqual(result['diaForcedSourceId'], 506428274000265224)
        self.assertEqual(result['diaObjectId'], 506428274000265713)

        with self.assertRaisesRegex(RuntimeError, "diaForcedSourceId=54321 not found"):
            self.apdb.load_forced_source(54321)

    def test_make_flag_exclusion_clause(self):
        # test clause generation with default flag list
        table = self.apdb._tables["DiaSource"]
        query = table.select()
        query = self.apdb._make_flag_exclusion_query(query, table, self.apdb.diaSource_flags_exclude)
        self.assertEqual(str(query.whereclause.compile(compile_kwargs={"literal_binds": True})),
                         '("DiaSource".flags & 972) = 0')

        with self.assertWarnsRegex(RuntimeWarning, "Flag bitmask is zero."):
            query = self.apdb._make_flag_exclusion_query(query, table, [])

    def test_set_excluded_diaSource_flags(self):
        with self.assertRaisesRegex(ValueError, "flag not a real flag not included"):
            self.apdb.set_excluded_diaSource_flags(['not a real flag'])

        self.apdb.set_excluded_diaSource_flags(['base_PixelFlags_flag'])
        table = self.apdb._tables["DiaSource"]
        query = table.select()
        query = self.apdb._make_flag_exclusion_query(query, table, self.apdb.diaSource_flags_exclude)
        self.assertEqual(str(query.whereclause.compile(compile_kwargs={"literal_binds": True})),
                         '("DiaSource".flags & 1) = 0')

    def test_fill_from_ccdVisitId(self):
        # an empty series should be unchanged
        empty = pd.Series()
        self.apdb._fill_from_ccdVisitId(empty)
        self.assertTrue(empty.equals(pd.Series()))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

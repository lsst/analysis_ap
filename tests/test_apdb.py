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

import lsst.utils.tests
import pandas as pd
from lsst.analysis.ap.apdb import ApdbSqliteQuery


class TestApdbSqlite(lsst.utils.tests.TestCase):
    def setUp(self):
        datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/")
        apdb_file = os.path.join(datadir, "apdb.sqlite3")
        self.apdb = ApdbSqliteQuery(apdb_file)

    def test_load_sources(self):
        result = self.apdb.load_sources(limit=None)
        self.assertEqual(len(result), 290)
        # spot check a few fields
        self.assertEqual(result['diaSourceId'][0], 506428274000265217)
        self.assertEqual(result['diaObjectId'][14], 506428274000265241)
        self.assertEqual(result['detector'][0], 168)
        self.assertEqual(result['visit'][0], 943296)

        # check using a query limit
        result = self.apdb.load_sources(limit=2)
        self.assertEqual(len(result), 2)

    def test_load_sources_exclude_flags(self):
        # Test that we load the expected number of diaSources.
        # (There are 19 diaSources of the 290 that should be excluded.)
        result = self.apdb.load_sources(exclude_flagged=True)
        self.assertEqual(len(result), 271)

    def test_load_sources_for_object(self):
        # Test that we load one specific diaObject and 1 of its 2 diaSources
        result = self.apdb.load_sources_for_object(506428274000265388)
        self.assertEqual(len(result), 2)
        self.assertEqual(result['diaSourceId'][0], 506428274000265388)

    def test_load_forced_sources_for_object(self):
        # Test that we can load the same diaObject
        # This diaObject was found to have 2 constituent diaForcedSources
        result = self.apdb.load_forced_sources_for_object(506428274000265388)
        self.assertEqual(len(result), 2)
        self.assertEqual(result['diaForcedSourceId'][0], 506428274000265354)

    def test_load_sources_for_object_exclude_flags(self):
        # diaObject chosen from inspection to have 2 flagged diaSources
        result = self.apdb.load_sources_for_object(506428274000265285)
        self.assertEqual(len(result), 2)
        self.assertEqual(result['diaSourceId'][0], 506428274000265285)
        self.assertEqual(result['diaSourceId'][1], 527736141479149663)
        # This same diaObject has `diaSource_flags_exclude` flags
        # on all 2 of its diaSources
        result = self.apdb.load_sources_for_object(506428274000265285,
                                                   exclude_flagged=True)
        self.assertEqual(len(result), 0)

    def test_load_objects(self):
        result = self.apdb.load_objects(limit=None)
        self.assertEqual(len(result), 259)
        # spot check a few fields
        self.assertNotIn("diaSourceId", result)
        self.assertEqual(result['diaObjectId'][0], 506428274000265217)
        self.assertIn("validityStart", result.columns)

        result = self.apdb.load_objects(limit=2)
        self.assertEqual(len(result), 2)

        # TODO DM-39503: No objects here have more than 1 source: once we have
        # a larger test APDB, implement this check.
        # result = self.apdb.load_objects(min_sources=2)
        # self.assertEqual(len(result), SOMETHING)

    def test_load_forced_sources(self):
        result = self.apdb.load_forced_sources(limit=None)
        self.assertEqual(len(result), 376)
        # spot check a few fields
        self.assertEqual(result['diaObjectId'][0], 506428274000265217)
        self.assertEqual(result['diaForcedSourceId'][0], 506428274000265217)
        self.assertEqual(result['detector'][0], 168)
        self.assertEqual(result['visit'][0], 943296)

        result = self.apdb.load_forced_sources(limit=2)
        self.assertEqual(len(result), 2)

    def test_load_source(self):
        result = self.apdb.load_source(506428274000265217)
        # spot check a few fields
        self.assertEqual(result['diaSourceId'], 506428274000265217)
        self.assertEqual(result['diaObjectId'], 506428274000265217)
        self.assertEqual(result['band'], 'r')

        with self.assertRaisesRegex(RuntimeError, "diaSourceId=54321 not found"):
            self.apdb.load_source(54321)

    def test_load_object(self):
        result = self.apdb.load_object(506428274000265228)
        # spot check a few fields
        self.assertEqual(result['diaObjectId'], 506428274000265228)
        self.assertFloatsAlmostEqual(result['ra'], 55.7887299103902, rtol=1e-15)

        with self.assertRaisesRegex(RuntimeError, "diaObjectId=54321 not found"):
            self.apdb.load_object(54321)

    def test_load_forced_source(self):
        result = self.apdb.load_forced_source(506428274000265224)
        # spot check a few fields
        self.assertEqual(result['diaForcedSourceId'], 506428274000265224)
        self.assertEqual(result['diaObjectId'], 506428274000265228)

        with self.assertRaisesRegex(RuntimeError, "diaForcedSourceId=54321 not found"):
            self.apdb.load_forced_source(54321)

    def test_make_flag_exclusion_clause(self):
        # Test clause generation with default flag list.
        table = self.apdb._tables["DiaSource"]
        query = table.select()
        query = self.apdb._make_flag_exclusion_query(query, table, self.apdb.diaSource_flags_exclude)
        # Check that the SQL query literal string does the flag exclusion.
        queryString = ('"DiaSource"."pixelFlags_bad" = false '
                       'AND "DiaSource"."pixelFlags_suspect" = false '
                       'AND "DiaSource"."pixelFlags_saturatedCenter" = false '
                       'AND "DiaSource"."pixelFlags_interpolated" = false '
                       'AND "DiaSource"."pixelFlags_interpolatedCenter" = false '
                       'AND "DiaSource"."pixelFlags_edge" = false')
        self.assertEqual(str(query.whereclause.compile(compile_kwargs={"literal_binds": True})),
                         queryString)

    def test_set_excluded_diaSource_flags(self):
        with self.assertRaisesRegex(ValueError, "flag not a real flag not included"):
            self.apdb.set_excluded_diaSource_flags(['not a real flag'])

        self.apdb.set_excluded_diaSource_flags(['pixelFlags_streak'])
        table = self.apdb._tables["DiaSource"]
        query = table.select()
        query = self.apdb._make_flag_exclusion_query(query, table, self.apdb.diaSource_flags_exclude)
        # Check that the SQL query does a non-default flag exclusion.
        queryString = '"DiaSource"."pixelFlags_streak" = false'
        self.assertEqual(str(query.whereclause.compile(compile_kwargs={"literal_binds": True})),
                         queryString)

    def test_fill_from_instrument(self):
        # an empty series should be unchanged
        empty = pd.Series()
        self.apdb._fill_from_instrument(empty)
        self.assertTrue(empty.equals(pd.Series()))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

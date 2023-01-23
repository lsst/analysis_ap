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

from lsst.obs.decam import DarkEnergyCamera
import lsst.daf.butler


class TestApdbSqlite(lsst.utils.tests.TestCase):
    def setUp(self):
        datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/")
        apdb_file = os.path.join(datadir, "apdb.sqlite3")

        # TODO: only necessary until we can get detector/visit from APDB
        self.path = tempfile.TemporaryDirectory()
        lsst.daf.butler.Butler.makeRepo(self.path.name)
        butler = lsst.daf.butler.Butler(self.path.name, writeable=True)
        DarkEnergyCamera().register(butler.registry, update=True)

        self.apdb = ApdbSqliteQuery(apdb_file, butler=butler, instrument="DECam")

    def tearDown(self):
        self.path.cleanup()

    def test_load_sources(self):
        sources = self.apdb.load_sources()
        self.assertEqual(len(sources), 15)
        # spot check a few fields
        self.assertEqual(sources['diaSourceId'][0], 224948952930189335)
        self.assertTrue(all(sources['ccdVisitId'] == 419000076))
        self.assertEqual(sources['diaObjectId'][14], 224948952930189349)
        self.assertEqual(sources['detector'][0], 76)
        self.assertEqual(sources['visit'][0], 4190000)

        # check using a query limit
        sources = self.apdb.load_sources(limit=2)
        self.assertEqual(len(sources), 2)

    def test_load_sources_exclude_flags(self):
        sources = self.apdb.load_sources(exclude_flagged=True)
        self.assertEqual(len(sources), 11)

    def test_load_sources_for_object(self):
        sources = self.apdb.load_sources_for_object(224948952930189335)
        # This test APDB has only one source per object.
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources['diaSourceId'][0], 224948952930189335)

    def test_load_sources_for_object_exclude_flags(self):
        # diaObjectId chosen from inspection to have flagged diaSources
        sources = self.apdb.load_sources_for_object(224948952930189342)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources['diaSourceId'][0], 224948952930189342)

        sources = self.apdb.load_sources_for_object(224948952930189342,
                                                    exclude_flagged=True)
        self.assertEqual(len(sources), 0)

    def test_load_objects(self):
        objects = self.apdb.load_objects()
        self.assertEqual(len(objects), 15)
        # spot check a few fields
        self.assertNotIn("diaSourceId", objects)
        self.assertEqual(objects['diaObjectId'][0], 224948952930189335)
        self.assertIn("validityStart", objects.columns)

        sources = self.apdb.load_objects(limit=2)
        self.assertEqual(len(sources), 2)

    def test_load_forced_sources(self):
        sources = self.apdb.load_forced_sources()
        self.assertEqual(len(sources), 15)
        # spot check a few fields
        self.assertEqual(sources['diaObjectId'][0], 224948952930189335)
        self.assertEqual(sources['diaForcedSourceId'][0], 224948952930189313)
        self.assertEqual(sources['detector'][0], 76)
        self.assertEqual(sources['visit'][0], 4190000)

        sources = self.apdb.load_forced_sources(limit=2)
        self.assertEqual(len(sources), 2)

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

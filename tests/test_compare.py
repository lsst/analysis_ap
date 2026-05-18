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

import astropy.units as u
import numpy as np
import pandas as pd

import lsst.utils.tests
from lsst.analysis.ap.compare import (
    match_catalogs, flux_residuals, match_to_truth,
)


def _make_frame(rows, **defaults):
    """Build a small DiaSource-like DataFrame from a list of partial dicts."""
    full = []
    for row in rows:
        merged = dict(defaults)
        merged.update(row)
        full.append(merged)
    return pd.DataFrame(full)


class TestMatchCatalogs(lsst.utils.tests.TestCase):
    """Spatial cross-matching is the building block for compare_sources;
    these tests pin down its behavior on small hand-crafted catalogs."""

    def setUp(self):
        # Three nearby points, two on (visit=1, detector=0), one on
        # (visit=1, detector=1). All separations are tiny (~0.1 arcsec).
        self.srcs1 = _make_frame([
            {"diaSourceId": 1, "ra": 10.0, "dec": -5.0, "visit": 1, "detector": 0},
            {"diaSourceId": 2, "ra": 10.001, "dec": -5.0, "visit": 1, "detector": 0},
            {"diaSourceId": 3, "ra": 11.0, "dec": -5.0, "visit": 1, "detector": 1},
        ])
        # srcs2: id 11 matches id 1 (same coord). id 12 is far from id 2.
        # No row in srcs2 for (visit=1, detector=1), so id 3 must be unique1.
        self.srcs2 = _make_frame([
            {"diaSourceId": 11, "ra": 10.0, "dec": -5.0, "visit": 1, "detector": 0},
            {"diaSourceId": 12, "ra": 10.5, "dec": -5.0, "visit": 1, "detector": 0},
        ])

    def test_basic_match(self):
        matched, unique1, unique2 = match_catalogs(
            self.srcs1, self.srcs2, radius=1*u.arcsec)
        # id=1 in srcs1 matches id=11 in srcs2.
        self.assertEqual(list(matched["diaSourceId"]), [1])
        self.assertEqual(list(matched["diaSourceId_2"]), [11])
        # The match is essentially zero arcsec apart.
        self.assertLess(float(matched["xmatch_dist_arcsec"].iloc[0]), 1e-6)
        # ids 2, 3 from srcs1 are unique; ids 12 from srcs2 is unique.
        self.assertEqual(set(unique1["diaSourceId"]), {2, 3})
        self.assertEqual(set(unique2["diaSourceId"]), {12})

    def test_radius_units(self):
        # Bare-float radius is interpreted as arcseconds.
        matched_q, _, _ = match_catalogs(self.srcs1, self.srcs2,
                                         radius=1*u.arcsec)
        matched_f, _, _ = match_catalogs(self.srcs1, self.srcs2, radius=1.0)
        self.assertEqual(len(matched_q), len(matched_f))

    def test_no_grouping(self):
        # With on=(), sources match across visit/detector boundaries.
        srcs1 = _make_frame([
            {"diaSourceId": 1, "ra": 10.0, "dec": 0.0, "visit": 1, "detector": 0},
        ])
        srcs2 = _make_frame([
            {"diaSourceId": 2, "ra": 10.0, "dec": 0.0, "visit": 99, "detector": 99},
        ])
        # With grouping, no match (different visits/detectors).
        matched, _, _ = match_catalogs(srcs1, srcs2, radius=1*u.arcsec)
        self.assertEqual(len(matched), 0)
        # Without grouping, the spatial match succeeds.
        matched, _, _ = match_catalogs(srcs1, srcs2, radius=1*u.arcsec, on=())
        self.assertEqual(len(matched), 1)

    def test_empty_inputs(self):
        empty = self.srcs1.iloc[0:0]
        matched, u1, u2 = match_catalogs(empty, self.srcs2, radius=1*u.arcsec)
        self.assertEqual(len(matched), 0)
        self.assertEqual(len(u1), 0)
        self.assertEqual(set(u2["diaSourceId"]), {11, 12})

        matched, u1, u2 = match_catalogs(self.srcs1, empty, radius=1*u.arcsec)
        self.assertEqual(len(matched), 0)
        self.assertEqual(set(u1["diaSourceId"]), {1, 2, 3})
        self.assertEqual(len(u2), 0)


class TestFluxResiduals(lsst.utils.tests.TestCase):
    """`flux_residuals` joins matched pairs with catalog 2 to compute
    flux differences in units of sigma."""

    def test_zero_residuals_when_identical(self):
        srcs1 = _make_frame([
            {"diaSourceId": 1, "ra": 10.0, "dec": 0.0, "visit": 1, "detector": 0,
             "psfFlux": 100.0, "psfFluxErr": 5.0},
            {"diaSourceId": 2, "ra": 10.001, "dec": 0.0, "visit": 1, "detector": 0,
             "psfFlux": 200.0, "psfFluxErr": 10.0},
        ])
        srcs2 = _make_frame([
            {"diaSourceId": 11, "ra": 10.0, "dec": 0.0, "visit": 1, "detector": 0,
             "psfFlux": 100.0, "psfFluxErr": 5.0},
            {"diaSourceId": 12, "ra": 10.001, "dec": 0.0, "visit": 1, "detector": 0,
             "psfFlux": 200.0, "psfFluxErr": 10.0},
        ])
        matched, _, _ = match_catalogs(srcs1, srcs2, radius=1*u.arcsec)
        residuals = flux_residuals(matched, srcs2)
        np.testing.assert_array_equal(residuals["delta_flux"], [0.0, 0.0])
        np.testing.assert_array_equal(residuals["delta_flux_sigma"], [0.0, 0.0])

    def test_known_offset(self):
        # Set up a 3-sigma flux offset on a single matched pair.
        srcs1 = _make_frame([
            {"diaSourceId": 1, "ra": 10.0, "dec": 0.0, "visit": 1, "detector": 0,
             "psfFlux": 130.0, "psfFluxErr": 5.0},
        ])
        srcs2 = _make_frame([
            {"diaSourceId": 11, "ra": 10.0, "dec": 0.0, "visit": 1, "detector": 0,
             "psfFlux": 100.0, "psfFluxErr": 5.0},
        ])
        matched, _, _ = match_catalogs(srcs1, srcs2, radius=1*u.arcsec)
        residuals = flux_residuals(matched, srcs2)
        self.assertEqual(float(residuals["delta_flux"].iloc[0]), 30.0)
        # Combined sigma is sqrt(5^2 + 5^2); 30/sqrt(50) ~ 4.243.
        self.assertAlmostEqual(
            float(residuals["delta_flux_sigma"].iloc[0]),
            30.0 / np.sqrt(50.0),
            places=10,
        )


class TestMatchToTruth(lsst.utils.tests.TestCase):
    """`match_to_truth` reports purity and completeness against a truth
    catalog by matching in both directions."""

    def setUp(self):
        # Three detected sources: two near truth, one in empty sky.
        self.srcs = _make_frame([
            {"diaSourceId": 1, "ra": 10.0, "dec": 0.0},
            {"diaSourceId": 2, "ra": 11.0, "dec": 0.0},
            {"diaSourceId": 3, "ra": 99.0, "dec": -50.0},  # bogus, not real
        ])
        # Three truth sources: two recovered, one missed.
        self.truth = _make_frame([
            {"injection_id": 100, "ra": 10.0, "dec": 0.0},
            {"injection_id": 101, "ra": 11.0, "dec": 0.0},
            {"injection_id": 102, "ra": 50.0, "dec": -10.0},  # not recovered
        ])

    def test_purity_and_completeness(self):
        result = match_to_truth(self.srcs, self.truth, radius=1*u.arcsec)
        # 2 of 3 detections are real; 2 of 3 truths are detected.
        self.assertAlmostEqual(result["purity"], 2/3, places=10)
        self.assertAlmostEqual(result["completeness"], 2/3, places=10)
        # The bogus detection has is_real=False and a NA partner.
        bogus = result["srcs"].set_index("diaSourceId").loc[3]
        self.assertFalse(bool(bogus["is_real"]))
        self.assertTrue(pd.isna(bogus["injection_id_match"]))
        # The unrecovered truth has detected=False.
        miss = result["truth"].set_index("injection_id").loc[102]
        self.assertFalse(bool(miss["detected"]))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

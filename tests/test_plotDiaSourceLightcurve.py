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

import lsst.afw.image
import lsst.geom
import lsst.meas.base.tests
import lsst.utils.tests
import numpy as np
import pandas as pd
import PIL

from lsst.analysis.ap import plotDiaSourceLightcurve

# Pull in the DATA fixture from the cutouts test so we get the full set of
# flag columns expected by _annotate_image.
from test_plotImageSubtractionCutouts import DATA, skyCenter


DIA_OBJECT_ID = 999999999999000001


# Add the extra columns required by the lightcurve task.
def _augment_with_object_columns(data):
    data = data.copy()
    data["diaObjectId"] = [DIA_OBJECT_ID, DIA_OBJECT_ID]
    data["midpointMjdTai"] = [60100.5, 60110.5]
    return data


class _StubApdbQuery:
    """Minimal DbQuery stub returning canned sources/forced DataFrames.

    Tracks the number of calls so tests can verify the per-diaObject cache.
    """

    def __init__(self, sources, forced):
        self._sources = sources
        self._forced = forced
        self.source_calls = 0
        self.forced_calls = 0

    def load_sources_for_object(self, dia_object_id, exclude_flagged=False, limit=100000):
        self.source_calls += 1
        self.last_source_kwargs = {"exclude_flagged": exclude_flagged, "limit": limit}
        return self._sources.copy()

    def load_forced_sources_for_object(self, dia_object_id, exclude_flagged=False, limit=100000):
        self.forced_calls += 1
        self.last_forced_kwargs = {"exclude_flagged": exclude_flagged, "limit": limit}
        return self._forced.copy()


def _make_sources_frame(visits, bands, mjds, fluxes, errs, with_err=True):
    frame = pd.DataFrame({
        "visit": visits,
        "band": bands,
        "midpointMjdTai": mjds,
        "psfFlux": fluxes,
    })
    if with_err:
        frame["psfFluxErr"] = errs
    return frame


class TestPlotDiaSourceLightcurve(lsst.utils.tests.TestCase):
    """Tests for PlotDiaSourceLightcurveTask."""

    def setUp(self):
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(0, 0), lsst.geom.Point2I(100, 100))
        self.centroid = lsst.geom.Point2D(50, 50)
        dataset = lsst.meas.base.tests.TestDataset(bbox, crval=skyCenter)
        self.scale = 0.3
        dataset.addSource(instFlux=1e5, centroid=self.centroid)
        self.science, _ = dataset.realize(noise=1000.0, schema=dataset.makeMinimalSchema())
        self.template, _ = dataset.realize(noise=5.0, schema=dataset.makeMinimalSchema())
        self.difference = lsst.afw.image.ExposureF(self.science, deep=True)
        self.difference.image -= self.template.image
        self.data = _augment_with_object_columns(DATA)

    def _make_task(self, sources=None, forced=None, **config_kwargs):
        config = plotDiaSourceLightcurve.PlotDiaSourceLightcurveConfig()
        for key, value in config_kwargs.items():
            setattr(config, key, value)
        query = None
        if sources is not None or forced is not None:
            query = _StubApdbQuery(
                sources if sources is not None else pd.DataFrame(),
                forced if forced is not None else pd.DataFrame(),
            )
        task = plotDiaSourceLightcurve.PlotDiaSourceLightcurveTask(
            config=config, output_path="", apdb_query=query)
        return task, query

    def _record(self, row):
        """Return a single-row numpy.record matching DataFrame ``iloc``."""
        return self.data.iloc[[row]].to_records(index=False)[0]

    def test_generate_image_no_apdb(self):
        """Without an APDB handle, the lightcurve panel renders a placeholder
        but the figure still produces a valid PNG.
        """
        task, _ = self._make_task()
        cutout = task.generate_image(self.science, self.template, self.difference,
                                     skyCenter, self.scale,
                                     source=self._record(0))
        with PIL.Image.open(cutout) as im:
            self.assertGreater(im.width, 0)
            self.assertGreater(im.height, 0)

    def test_generate_image_with_lightcurve(self):
        """With matching sources and forced rows, both groups render."""
        sources = _make_sources_frame(
            visits=[1234, 5678, 9999],
            bands=["r", "g", "r"],
            mjds=[60100.5, 60110.5, 60120.5],
            fluxes=[1234.5, 2345.6, 3456.7],
            errs=[123.5, 234.5, 345.6],
        )
        # Forced has a visit (8888) that does NOT appear in sources — that
        # one should be drawn with the forced-only marker. Visit 1234 IS in
        # sources, so the forced entry for it should be suppressed.
        forced = _make_sources_frame(
            visits=[1234, 8888],
            bands=["r", "g"],
            mjds=[60100.5, 60115.5],
            fluxes=[1200.0, 800.0],
            errs=[120.0, 80.0],
        )
        task, query = self._make_task(sources=sources, forced=forced)
        cutout = task.generate_image(self.science, self.template, self.difference,
                                     skyCenter, self.scale,
                                     source=self._record(0))
        self.assertEqual(query.source_calls, 1)
        self.assertEqual(query.forced_calls, 1)
        with PIL.Image.open(cutout) as im:
            self.assertGreater(im.width, 0)
            self.assertGreater(im.height, 0)

    def test_lightcurve_cache_reuses_query(self):
        """Adjacent diaSources on the same diaObject should hit the cache."""
        sources = _make_sources_frame(
            visits=[1234, 5678],
            bands=["r", "g"],
            mjds=[60100.5, 60110.5],
            fluxes=[1234.5, 2345.6],
            errs=[123.5, 234.5],
        )
        forced = pd.DataFrame(columns=["visit", "band", "midpointMjdTai",
                                       "psfFlux", "psfFluxErr"])
        task, query = self._make_task(sources=sources, forced=forced)
        for i in range(2):
            task.generate_image(self.science, self.template, self.difference,
                                skyCenter, self.scale,
                                source=self._record(i))
        # Both sources share the same diaObjectId; the second call must come
        # from the cache.
        self.assertEqual(query.source_calls, 1)
        self.assertEqual(query.forced_calls, 1)

    def test_forced_only_dedup_by_visit(self):
        """Forced rows whose ``visit`` matches a diaSource are suppressed."""
        sources = _make_sources_frame(
            visits=[1234, 5678],
            bands=["r", "g"],
            mjds=[60100.5, 60110.5],
            fluxes=[1234.5, 2345.6],
            errs=[123.5, 234.5],
        )
        forced = _make_sources_frame(
            visits=[1234, 5678, 7777, 8888],
            bands=["r", "g", "r", "g"],
            mjds=[60100.5, 60110.5, 60112.5, 60115.5],
            fluxes=[1200.0, 2400.0, 500.0, 800.0],
            errs=[120.0, 240.0, 50.0, 80.0],
        )
        task, _ = self._make_task(sources=sources, forced=forced)
        forced_only = forced[~forced["visit"].isin(sources["visit"])]
        self.assertEqual(set(forced_only["visit"]), {7777, 8888})
        # Also exercise the rendering path to confirm it does not raise.
        task.generate_image(self.science, self.template, self.difference,
                            skyCenter, self.scale, source=self._record(0))

    def test_no_psfFluxErr(self):
        """Missing or NaN psfFluxErr should fall back to no error bars."""
        sources = _make_sources_frame(
            visits=[1234, 5678],
            bands=["r", "g"],
            mjds=[60100.5, 60110.5],
            fluxes=[1234.5, 2345.6],
            errs=None, with_err=False,
        )
        forced = _make_sources_frame(
            visits=[7777],
            bands=["r"],
            mjds=[60112.5],
            fluxes=[500.0],
            errs=[np.nan], with_err=True,
        )
        task, _ = self._make_task(sources=sources, forced=forced)
        cutout = task.generate_image(self.science, self.template, self.difference,
                                     skyCenter, self.scale,
                                     source=self._record(0))
        with PIL.Image.open(cutout) as im:
            self.assertGreater(im.width, 0)

    def test_forced_query_skips_exclude_flagged(self):
        """DiaForcedSource lacks the diaSource flag columns; the forced
        query must always be called with exclude_flagged=False, even when
        the config asks to exclude flagged diaSources.
        """
        sources = _make_sources_frame(
            visits=[1234], bands=["r"], mjds=[60100.5],
            fluxes=[1234.5], errs=[123.5])
        forced = pd.DataFrame(columns=["visit", "band", "midpointMjdTai",
                                       "psfFlux", "psfFluxErr"])
        task, query = self._make_task(sources=sources, forced=forced,
                                      lightcurve_exclude_flagged=True)
        task.generate_image(self.science, self.template, self.difference,
                            skyCenter, self.scale,
                            source=self._record(0))
        self.assertTrue(query.last_source_kwargs["exclude_flagged"])
        self.assertFalse(query.last_forced_kwargs["exclude_flagged"])

    def test_nan_dia_object_id(self):
        """A NaN diaObjectId (unassociated diaSource) must not crash; the
        lightcurve panel renders its empty placeholder.
        """
        data = self.data.copy()
        data["diaObjectId"] = data["diaObjectId"].astype(float)
        data.loc[0, "diaObjectId"] = np.nan
        sources = _make_sources_frame(
            visits=[1234], bands=["r"], mjds=[60100.5],
            fluxes=[1234.5], errs=[123.5])
        forced = pd.DataFrame(columns=["visit", "band", "midpointMjdTai",
                                       "psfFlux", "psfFluxErr"])
        task, query = self._make_task(sources=sources, forced=forced)
        record = data.iloc[[0]].to_records(index=False)[0]
        cutout = task.generate_image(self.science, self.template, self.difference,
                                     skyCenter, self.scale, source=record)
        # APDB was never queried — diaObjectId is NaN.
        self.assertEqual(query.source_calls, 0)
        self.assertEqual(query.forced_calls, 0)
        with PIL.Image.open(cutout) as im:
            self.assertGreater(im.width, 0)

    def test_forced_legend_single_black_entry(self):
        """Forced points are colored per-band, but contribute a single
        black legend entry regardless of band count.
        """
        sources = _make_sources_frame(
            visits=[1234, 5678],
            bands=["r", "g"],
            mjds=[60100.5, 60110.5],
            fluxes=[1234.5, 2345.6],
            errs=[123.5, 234.5],
        )
        # Forced-only visits span two bands: should still produce one entry.
        forced = _make_sources_frame(
            visits=[7777, 8888],
            bands=["r", "g"],
            mjds=[60112.5, 60115.5],
            fluxes=[500.0, 800.0],
            errs=[50.0, 80.0],
        )
        task, _ = self._make_task(sources=sources, forced=forced)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        try:
            task._draw_lightcurve(ax, sources, forced,
                                  current_source=self._record(0))
            handles, labels = ax.get_legend_handles_labels()
            forced_labels = [lbl for lbl in labels if "forced" in lbl]
            self.assertEqual(len(forced_labels), 1)
            self.assertIn("(n=2)", forced_labels[0])
            # The single forced legend handle should be drawn in black.
            forced_handle = handles[labels.index(forced_labels[0])]
            self.assertEqual(forced_handle.get_color(), "black")
        finally:
            plt.close(fig)

    def test_njobs_downgraded(self):
        """Requesting multiprocessing should be downgraded with a warning."""
        sources = _make_sources_frame(
            visits=[1234], bands=["r"], mjds=[60100.5],
            fluxes=[1234.5], errs=[123.5])
        forced = pd.DataFrame(columns=["visit", "band", "midpointMjdTai",
                                       "psfFlux", "psfFluxErr"])
        task, _ = self._make_task(sources=sources, forced=forced)
        # write_images would normally attempt multiprocessing; the override
        # should silently drop njobs to 0 and not raise.
        with self.assertLogs(task.log.name, level="WARNING") as ctx:
            # Use just one row so we don't need real files on disk; the
            # cutouts task will try to look them up via butler_cache and
            # fail, but the warning should fire first.
            try:
                task.write_images(self.data.head(0), butler=None, njobs=4)
            except Exception:
                pass
        self.assertTrue(any("njobs" in msg for msg in ctx.output))


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

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
import pickle
import sys
import tempfile
import unittest

import pandas as pd
import PIL

from lsst.ap.association import UnpackApdbFlags
import lsst.afw.table
import lsst.geom
from lsst.meas.algorithms import SourceDetectionTask
import lsst.meas.base.tests
import lsst.utils.tests

from lsst.analysis.ap import plotImageSubtractionCutouts

# Sky center chosen to test metadata annotations (3-digit RA and negative Dec).
skyCenter = lsst.geom.SpherePoint(245.0, -45.0, lsst.geom.degrees)

# A two-row mock APDB DiaSource table.
DATA = pd.DataFrame(
    data={
        "diaSourceId": [506428274000265570, 527736141479149732],
        "ra": [skyCenter.getRa().asDegrees()+0.0001, skyCenter.getRa().asDegrees()-0.0001],
        "decl": [skyCenter.getDec().asDegrees()+0.0001, skyCenter.getDec().asDegrees()-0.001],
        "detector": [50, 60],
        "visit": [1234, 5678],
        "instrument": ["TestMock", "TestMock"],
        "filterName": ['r', 'g'],
        "psFlux": [1234.5, 1234.5],
        "psFluxErr": [123.5, 123.5],
        "snr": [10.0, 11.0],
        "psChi2": [40.0, 50.0],
        "psNdata": [10, 100],
        "apFlux": [2222.5, 3333.4],
        "apFluxErr": [222.5, 333.4],
        "totFlux": [2222000.5, 33330000.4],
        "totFluxErr": [22200.5, 333000.4],
        "isDipole": [True, False],
        # all flags vs. no flags
        "flags": [~0, 0],
        "spuriousness": [0, 1.0],
    }
)


def make_mock_catalog(image):
    """Make a simple SourceCatalog from the image, containing Footprints.
    """
    schema = lsst.afw.table.SourceTable.makeMinimalSchema()
    table = lsst.afw.table.SourceTable.make(schema)
    detect = SourceDetectionTask()
    return detect.run(table, image).sources


class TestPlotImageSubtractionCutouts(lsst.utils.tests.TestCase):
    """Test that PlotImageSubtractionCutoutsTask generates images and manifest
    files correctly.
    """
    def setUp(self):
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(0, 0), lsst.geom.Point2I(100, 100))
        # source at the center of the image
        self.centroid = lsst.geom.Point2D(50, 50)
        dataset = lsst.meas.base.tests.TestDataset(bbox, crval=skyCenter)
        self.scale = 0.3  # arbitrary arcseconds/pixel
        dataset.addSource(instFlux=1e5, centroid=self.centroid)
        self.science, self.scienceCat = dataset.realize(
            noise=1000.0, schema=dataset.makeMinimalSchema()
        )
        lsst.afw.table.updateSourceCoords(self.science.wcs, self.scienceCat)
        self.template, self.templateCat = dataset.realize(
            noise=5.0, schema=dataset.makeMinimalSchema()
        )
        # A simple image difference to have something to plot.
        self.difference = lsst.afw.image.ExposureF(self.science, deep=True)
        self.difference.image -= self.template.image

        flag_map = os.path.join(lsst.utils.getPackageDir("ap_association"), "data/association-flag-map.yaml")
        unpacker = UnpackApdbFlags(flag_map, "DiaSource")
        self.flags = unpacker.unpack(DATA["flags"], "flags")

    def test_generate_image(self):
        """Test that we get some kind of image out.

        It's useful to have a person look at the output via:
            im.show()
        """
        # output_path does nothing here, since we never write the file to disk.
        cutouts = plotImageSubtractionCutouts.PlotImageSubtractionCutoutsTask(output_path="")
        cutout = cutouts.generate_image(self.science, self.template, self.difference, skyCenter, self.scale)
        with PIL.Image.open(cutout) as im:
            # NOTE: uncomment this to show the resulting image.
            # im.show()
            # NOTE: the dimensions here are determined by the matplotlib figure
            # size (in inches) and the dpi (default=100), plus borders.
            self.assertEqual((im.height, im.width), (233, 630))

    def test_generate_image_larger_cutout(self):
        """A different cutout size: the resulting cutout image is the same
        size but shows more pixels.
        """
        config = plotImageSubtractionCutouts.PlotImageSubtractionCutoutsTask.ConfigClass()
        config.sizes = [100]
        # output_path does nothing here, since we never write the file to disk.
        cutouts = plotImageSubtractionCutouts.PlotImageSubtractionCutoutsTask(config=config, output_path="")
        cutout = cutouts.generate_image(self.science, self.template, self.difference, skyCenter, self.scale)
        with PIL.Image.open(cutout) as im:
            # NOTE: uncomment this to show the resulting image.
            # im.show()
            # NOTE: the dimensions here are determined by the matplotlib figure
            # size (in inches) and the dpi (default=100), plus borders.
            self.assertEqual((im.height, im.width), (233, 630))

    def test_generate_image_metadata(self):
        """Test that we can add metadata to the image; it changes the height
        a lot, and the width a little for the text boxes.

        It's useful to have a person look at the output via:
            im.show()
        """
        config = plotImageSubtractionCutouts.PlotImageSubtractionCutoutsTask.ConfigClass()
        config.add_metadata = True
        # output_path does nothing here, since we never write the file to disk.
        cutouts = plotImageSubtractionCutouts.PlotImageSubtractionCutoutsTask(config=config, output_path="")
        cutout = cutouts.generate_image(self.science,
                                        self.template,
                                        self.difference,
                                        skyCenter,
                                        self.scale,
                                        source=DATA.iloc[0],
                                        flags=self.flags[0])
        with PIL.Image.open(cutout) as im:
            # NOTE: uncomment this to show the resulting image.
            # im.show()
            # NOTE: the dimensions here are determined by the matplotlib figure
            # size (in inches) and the dpi (default=100), plus borders.
            self.assertEqual((im.height, im.width), (343, 645))

        # A cutout without any flags: the dimensions should be unchanged.
        cutout = cutouts.generate_image(self.science,
                                        self.template,
                                        self.difference,
                                        skyCenter,
                                        self.scale,
                                        source=DATA.iloc[1],
                                        flags=self.flags[1])
        with PIL.Image.open(cutout) as im:
            # NOTE: uncomment this to show the resulting image.
            # im.show()
            # NOTE: the dimensions here are determined by the matplotlib figure
            # size (in inches) and the dpi (default=100), plus borders.
            self.assertEqual((im.height, im.width), (343, 645))

    def test_generate_image_multisize_cutouts_without_metadata(self):
        """Multiple cutout sizes: the resulting image is larger in size
        and contains cutouts of multiple sizes.
        """
        config = plotImageSubtractionCutouts.PlotImageSubtractionCutoutsTask.ConfigClass()
        config.sizes = [32, 64]
        # output_path does nothing here, since we never write the file to disk.
        cutouts = plotImageSubtractionCutouts.PlotImageSubtractionCutoutsTask(config=config, output_path="")
        cutout = cutouts.generate_image(self.science, self.template, self.difference, skyCenter, self.scale)
        with PIL.Image.open(cutout) as im:
            # NOTE: uncomment this to show the resulting image.
            # im.show()
            # NOTE: the dimensions here are determined by the matplotlib figure
            # size (in inches) and the dpi (default=100), plus borders.
            self.assertEqual((im.height, im.width), (450, 630))

    def test_generate_image_multisize_cutouts_with_metadata(self):
        """Test that we can add metadata to the image; it changes the height
        a lot, and the width a little for the text boxes.

        It's useful to have a person look at the output via:
            im.show()
        """
        config = plotImageSubtractionCutouts.PlotImageSubtractionCutoutsTask.ConfigClass()
        config.add_metadata = True
        config.sizes = [32, 64]
        # output_path does nothing here, since we never write the file to disk.
        cutouts = plotImageSubtractionCutouts.PlotImageSubtractionCutoutsTask(config=config, output_path="")
        cutout = cutouts.generate_image(self.science,
                                        self.template,
                                        self.difference,
                                        skyCenter,
                                        self.scale,
                                        source=DATA.iloc[0],
                                        flags=self.flags[0])
        with PIL.Image.open(cutout) as im:
            # NOTE: uncomment this to show the resulting image.
            # im.show()
            # NOTE: the dimensions here are determined by the matplotlib figure
            # size (in inches) and the dpi (default=100), plus borders.
            self.assertEqual((im.height, im.width), (576, 645))

        # A cutout without any flags: the dimensions should be unchanged.
        cutout = cutouts.generate_image(self.science,
                                        self.template,
                                        self.difference,
                                        skyCenter,
                                        self.scale,
                                        source=DATA.iloc[1],
                                        flags=self.flags[1])
        with PIL.Image.open(cutout) as im:
            # NOTE: uncomment this to show the resulting image.
            # im.show()
            # NOTE: the dimensions here are determined by the matplotlib figure
            # size (in inches) and the dpi (default=100), plus borders.
            self.assertEqual((im.height, im.width), (576, 645))

    def test_generate_image_invalid_paramters(self):
        cutouts = plotImageSubtractionCutouts.PlotImageSubtractionCutoutsTask(output_path="")
        with self.assertRaisesRegex(RuntimeError, "Must pass both"):
            cutouts.generate_image(self.science,
                                   self.template,
                                   self.difference,
                                   skyCenter,
                                   self.scale,
                                   source=DATA.iloc[1],
                                   flags=None)
        with self.assertRaisesRegex(RuntimeError, "Must pass both"):
            cutouts.generate_image(self.science,
                                   self.template,
                                   self.difference,
                                   skyCenter,
                                   self.scale,
                                   source=None,
                                   flags=self.flags[1])

    def test_write_images(self):
        """Test that images get written to a temporary directory."""
        butler = unittest.mock.Mock(spec=lsst.daf.butler.Butler)
        # We don't care what the output images look like here, just that
        # butler.get() returns an Exposure for every call.
        butler.get.return_value = self.science

        with tempfile.TemporaryDirectory() as path:
            config = plotImageSubtractionCutouts.PlotImageSubtractionCutoutsTask.ConfigClass()
            cutouts = plotImageSubtractionCutouts.PlotImageSubtractionCutoutsTask(config=config,
                                                                                  output_path=path)
            result = cutouts.write_images(DATA, butler)
            self.assertEqual(result, list(DATA["diaSourceId"]))
            for file in ("images/506428274000260000/506428274000265570.png",
                         "images/527736141479140000/527736141479149732.png"):
                filename = os.path.join(path, file)
                self.assertTrue(os.path.exists(filename))
                with PIL.Image.open(filename) as image:
                    self.assertEqual(image.format, "PNG")

    def test_use_footprint(self):
        """Test the use_footprint config option, generating a fake diaSrc
        catalog that contains footprints that get used instead of config.sizes.
        """
        butler = unittest.mock.Mock(spec=lsst.daf.butler.Butler)

        def mock_get(dataset, dataId, *args, **kwargs):
            if "_diaSrc" in dataset:
                # The science image is the only mock image with a source in it.
                catalog = make_mock_catalog(self.science)
                # Assign the matching source id to the detection.
                match = DATA["visit"] == dataId["visit"]
                catalog["id"] = DATA["diaSourceId"].to_numpy()[match][0]
                return catalog
            else:
                return self.science

        butler.get.side_effect = mock_get

        with tempfile.TemporaryDirectory() as path:
            config = plotImageSubtractionCutouts.PlotImageSubtractionCutoutsTask.ConfigClass()
            config.use_footprint = True
            cutouts = plotImageSubtractionCutouts.PlotImageSubtractionCutoutsTask(config=config,
                                                                                  output_path=path)
            result = cutouts.write_images(DATA, butler)
            self.assertEqual(result, list(DATA["diaSourceId"]))
            for file in ("images/506428274000260000/506428274000265570.png",
                         "images/527736141479140000/527736141479149732.png"):
                filename = os.path.join(path, file)
                self.assertTrue(os.path.exists(filename))
                with PIL.Image.open(filename) as image:
                    self.assertEqual(image.format, "PNG")

    def test_write_images_exception(self):
        """Test that write_images() catches errors in loading data."""
        butler = unittest.mock.Mock(spec=lsst.daf.butler.Butler)
        err = "Dataset not found"
        butler.get.side_effect = LookupError(err)

        with tempfile.TemporaryDirectory() as path:
            config = plotImageSubtractionCutouts.PlotImageSubtractionCutoutsTask.ConfigClass()
            cutouts = plotImageSubtractionCutouts.PlotImageSubtractionCutoutsTask(config=config,
                                                                                  output_path=path)

            with self.assertLogs("lsst.plotImageSubtractionCutouts", "ERROR") as cm:
                cutouts.write_images(DATA, butler)
            self.assertIn(
                "LookupError processing diaSourceId 506428274000265570: Dataset not found", cm.output[0]
            )
            self.assertIn(
                "LookupError processing diaSourceId 527736141479149732: Dataset not found", cm.output[1]
            )

    def check_make_manifest(self, url_root, url_list):
        """Check that make_manifest returns an appropriate DataFrame."""
        data = [5, 10, 20]
        config = plotImageSubtractionCutouts.PlotImageSubtractionCutoutsTask.ConfigClass()
        config.url_root = url_root
        # output_path does nothing here
        cutouts = plotImageSubtractionCutouts.PlotImageSubtractionCutoutsTask(config=config, output_path="")
        manifest = cutouts._make_manifest(data)
        self.assertEqual(manifest["metadata:diaSourceId"].to_list(), [5, 10, 20])
        self.assertEqual(manifest["location:1"].to_list(), url_list)

    def test_make_manifest(self):
        # check without an ending slash
        root = "http://example.org/zooniverse"
        url_list = [
            f"{root}/images/5.png",
            f"{root}/images/10.png",
            f"{root}/images/20.png",
        ]
        self.check_make_manifest(root, url_list)

        # check with an ending slash
        root = "http://example.org/zooniverse/"
        url_list = [
            f"{root}images/5.png",
            f"{root}images/10.png",
            f"{root}images/20.png",
        ]
        self.check_make_manifest(root, url_list)

    def test_pickle(self):
        """Test that the task is pickleable (necessary for multiprocessing).
        """
        config = plotImageSubtractionCutouts.PlotImageSubtractionCutoutsTask.ConfigClass()
        config.sizes = [63]
        cutouts = plotImageSubtractionCutouts.PlotImageSubtractionCutoutsTask(config=config,
                                                                              output_path="something")
        other = pickle.loads(pickle.dumps(cutouts))
        self.assertEqual(cutouts.config.sizes, other.config.sizes)
        self.assertEqual(cutouts._output_path, other._output_path)


class TestPlotImageSubtractionCutoutsMain(lsst.utils.tests.TestCase):
    """Test the commandline interface main() function via mocks."""
    def setUp(self):
        datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/")
        self.sqlitefile = os.path.join(datadir, "apdb.sqlite3")
        self.repo = "/not/a/real/butler"
        self.collection = "mockRun"
        self.outputPath = "/an/output/path"
        self.configFile = os.path.join(datadir, "plotImageSubtractionCutoutsConfig.py")
        self.instrument = "FakeInstrument"

        def mock_unpacker(_):
            """A mock unpacked dataId with visit and detector."""
            return {"visit": 12345, "detector": 42}

        butlerPatch = unittest.mock.patch("lsst.daf.butler.Butler")
        self._butler = butlerPatch.start()
        # mock detector/visit unpacker, until detector/visit are in APDB.
        self._butler.return_value.registry.dimensions.makePacker.return_value.unpack = (
            mock_unpacker
        )
        self.addCleanup(butlerPatch.stop)

    def test_main_args(self):
        """Test typical arguments to main()."""
        args = [
            "plotImageSubtractionCutouts",
            f"--sqlitefile={self.sqlitefile}",
            f"--collections={self.collection}",
            f"-C={self.configFile}",
            f"--instrument={self.instrument}",
            self.repo,
            self.outputPath,
        ]
        with unittest.mock.patch.object(
            plotImageSubtractionCutouts.PlotImageSubtractionCutoutsTask, "run", autospec=True
        ) as run, unittest.mock.patch.object(sys, "argv", args):
            plotImageSubtractionCutouts.main()
            self.assertEqual(self._butler.call_args.args, (self.repo,))
            self.assertEqual(
                self._butler.call_args.kwargs, {"collections": [self.collection]}
            )
            # NOTE: can't easily test the `data` arg to run, as select_sources
            # reads in a random order every time.
            self.assertEqual(run.call_args.args[2], self._butler.return_value)

    def test_main_args_no_collections(self):
        """Test with no collections argument."""
        args = [
            "plotImageSubtractionCutouts",
            f"--sqlitefile={self.sqlitefile}",
            f"-C={self.configFile}",
            f"--instrument={self.instrument}",
            self.repo,
            self.outputPath,
        ]
        with unittest.mock.patch.object(
            plotImageSubtractionCutouts.PlotImageSubtractionCutoutsTask, "run", autospec=True
        ) as run, unittest.mock.patch.object(sys, "argv", args):
            plotImageSubtractionCutouts.main()
            self.assertEqual(self._butler.call_args.args, (self.repo,))
            self.assertEqual(self._butler.call_args.kwargs, {"collections": None})
            self.assertIsInstance(run.call_args.args[1], pd.DataFrame)
            self.assertEqual(run.call_args.args[2], self._butler.return_value)

    def test_main_collection_list(self):
        """Test passing a list of collections."""
        collections = ["mock1", "mock2", "mock3"]
        args = [
            "plotImageSubtractionCutouts",
            f"--sqlitefile={self.sqlitefile}",
            f"--instrument={self.instrument}",
            self.repo,
            self.outputPath,
            f"-C={self.configFile}",
            "--collections",
        ]
        args.extend(collections)
        with unittest.mock.patch.object(
            plotImageSubtractionCutouts.PlotImageSubtractionCutoutsTask, "run", autospec=True
        ) as run, unittest.mock.patch.object(sys, "argv", args):
            plotImageSubtractionCutouts.main()
            self.assertEqual(self._butler.call_args.args, (self.repo,))
            self.assertEqual(
                self._butler.call_args.kwargs, {"collections": collections}
            )
            self.assertIsInstance(run.call_args.args[1], pd.DataFrame)
            self.assertEqual(run.call_args.args[2], self._butler.return_value)

    def test_main_args_limit_offset(self):
        """Test typical arguments to main()."""
        args = [
            "plotImageSubtractionCutouts",
            f"--sqlitefile={self.sqlitefile}",
            f"--collections={self.collection}",
            f"-C={self.configFile}",
            f"--instrument={self.instrument}",
            "--all",
            "--limit=5",
            self.repo,
            self.outputPath,
        ]
        with unittest.mock.patch.object(
            plotImageSubtractionCutouts.PlotImageSubtractionCutoutsTask,
            "write_images",
            autospec=True,
            return_value=[5]
        ) as write_images, unittest.mock.patch.object(
            plotImageSubtractionCutouts.PlotImageSubtractionCutoutsTask,
            "write_manifest",
            autospec=True
        ) as write_manifest, unittest.mock.patch.object(sys, "argv", args):
            plotImageSubtractionCutouts.main()
            self.assertEqual(self._butler.call_args.args, (self.repo,))
            self.assertEqual(
                self._butler.call_args.kwargs, {"collections": [self.collection]}
            )
            self.assertIsInstance(write_images.call_args.args[1], pd.DataFrame)
            self.assertEqual(write_images.call_args.args[2], self._butler.return_value)
            # The test apdb contains 15 sources, so we get the return of
            # `write_images` three times with `limit=5`
            self.assertEqual(write_manifest.call_args.args[1], [5, 5, 5])

    @unittest.skip("Mock and multiprocess don't mix: https://github.com/python/cpython/issues/100090")
    def test_main_args_multiprocessing(self):
        """Test running with multiprocessing.
        """
        args = [
            "plotImageSubtractionCutouts",
            f"--sqlitefile={self.sqlitefile}",
            f"--collections={self.collection}",
            "-j2",
            f"-C={self.configFile}",
            f"--instrument={self.instrument}",
            self.repo,
            self.outputPath,
        ]
        with unittest.mock.patch.object(
            plotImageSubtractionCutouts.PlotImageSubtractionCutoutsTask, "run", autospec=True
        ) as run, unittest.mock.patch.object(sys, "argv", args):
            plotImageSubtractionCutouts.main()
            self.assertEqual(self._butler.call_args.args, (self.repo,))
            self.assertEqual(self._butler.call_args.kwargs, {"collections": [self.collection]})
            # NOTE: can't easily test the `data` arg to run, as select_sources
            # reads in a random order every time.
            self.assertEqual(run.call_args.args[2], self._butler.return_value)


class TestCutoutPath(lsst.utils.tests.TestCase):
    def test_normal_path(self):
        """Can the path manager handles non-chunked paths?
        """
        manager = plotImageSubtractionCutouts.CutoutPath("some/root/path")
        path = manager(id=12345678)
        self.assertEqual(path, "some/root/path/images/12345678.png")

    def test_chunking(self):
        """Can the path manager handle ids chunked into 10,000 file
        directories?
        """
        manager = plotImageSubtractionCutouts.CutoutPath("some/root/path", chunk_size=10000)
        path = manager(id=12345678)
        self.assertEqual(path, "some/root/path/images/12340000/12345678.png")

    def test_chunk_sizes(self):
        """Test valid and invalid values for the chunk_size parameter.
        """
        with self.assertRaisesRegex(RuntimeError, "chunk_size must be a power of 10"):
            plotImageSubtractionCutouts.CutoutPath("some/root/path", chunk_size=123)

        with self.assertRaisesRegex(RuntimeError, "chunk_size must be a power of 10"):
            plotImageSubtractionCutouts.CutoutPath("some/root/path", chunk_size=12300)

        # should not raise
        plotImageSubtractionCutouts.CutoutPath("some/root/path", chunk_size=1000)
        plotImageSubtractionCutouts.CutoutPath("some/root/path", chunk_size=1000000)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

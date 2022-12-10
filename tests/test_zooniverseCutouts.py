# This file is part of pipe_tasks.
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
import lsst.meas.base.tests
import lsst.utils.tests

from lsst.analysis.ap import zooniverseCutouts

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
    }
)


class TestZooniverseCutouts(lsst.utils.tests.TestCase):
    """Test that ZooniverseCutoutsTask generates images and manifest files
    correctly.
    """

    def setUp(self):
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(0, 0), lsst.geom.Point2I(100, 100))
        # source at the center of the image
        self.centroid = lsst.geom.Point2D(50, 50)
        dataset = lsst.meas.base.tests.TestDataset(bbox, crval=skyCenter)
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
        cutouts = zooniverseCutouts.ZooniverseCutoutsTask()
        cutout = cutouts.generate_image(
            self.science, self.template, self.difference, skyCenter
        )
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
        config = zooniverseCutouts.ZooniverseCutoutsTask.ConfigClass()
        config.size = 100
        cutouts = zooniverseCutouts.ZooniverseCutoutsTask(config=config)
        cutout = cutouts.generate_image(
            self.science, self.template, self.difference, skyCenter
        )
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
        config = zooniverseCutouts.ZooniverseCutoutsTask.ConfigClass()
        config.addMetadata = True
        cutouts = zooniverseCutouts.ZooniverseCutoutsTask(config=config)
        cutout = cutouts.generate_image(self.science,
                                        self.template,
                                        self.difference,
                                        skyCenter,
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
                                        source=DATA.iloc[1],
                                        flags=self.flags[1])
        with PIL.Image.open(cutout) as im:
            # NOTE: uncomment this to show the resulting image.
            # im.show()
            # NOTE: the dimensions here are determined by the matplotlib figure
            # size (in inches) and the dpi (default=100), plus borders.
            self.assertEqual((im.height, im.width), (343, 645))

    def test_write_images(self):
        """Test that images get written to a temporary directory."""
        butler = unittest.mock.Mock(spec=lsst.daf.butler.Butler)
        # We don't care what the output images look like here, just that
        # butler.get() returns an Exposure for every call.
        butler.get.return_value = self.science

        with tempfile.TemporaryDirectory() as path:
            config = zooniverseCutouts.ZooniverseCutoutsTask.ConfigClass()
            cutouts = zooniverseCutouts.ZooniverseCutoutsTask(config=config)
            result = cutouts.write_images(DATA, butler, path)
            self.assertEqual(result, list(DATA["diaSourceId"]))
            for file in ("images/506428274000265570.png", "images/527736141479149732.png"):
                filename = os.path.join(path, file)
                self.assertTrue(os.path.exists(filename))
                with PIL.Image.open(filename) as image:
                    self.assertEqual(image.format, "PNG")

    @unittest.skip("Mock and multiprocess don't mix: https://github.com/python/cpython/issues/100090")
    def test_write_images_multiprocess(self):
        """Test that images get written when multiprocessing is on."""
        butler = unittest.mock.Mock(spec=lsst.daf.butler.Butler)
        # We don't care what the output images look like here, just that
        # butler.get() returns an Exposure for every call.
        butler.get.return_value = self.science
        # Override __reduce__ to allow this mock to be pickleable.
        state = {"_mock_children": butler._mock_children}
        butler.__reduce__ = lambda self: (unittest.mock.Mock, (), state)

        with tempfile.TemporaryDirectory() as path:
            config = zooniverseCutouts.ZooniverseCutoutsTask.ConfigClass()
            config.n_processes = 2
            cutouts = zooniverseCutouts.ZooniverseCutoutsTask(config=config)
            result = cutouts.write_images(DATA, butler, path)
            self.assertEqual(result, list(DATA["diaSourceId"]))
            for file in ("images/506428274000265570.png", "images/527736141479149732.png"):
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
            config = zooniverseCutouts.ZooniverseCutoutsTask.ConfigClass()
            cutouts = zooniverseCutouts.ZooniverseCutoutsTask(config=config)

            with self.assertLogs("lsst.zooniverseCutouts", "ERROR") as cm:
                cutouts.write_images(DATA, butler, path)
            self.assertIn(
                "LookupError processing diaSourceId 506428274000265570: Dataset not found", cm.output[0]
            )
            self.assertIn(
                "LookupError processing diaSourceId 527736141479149732: Dataset not found", cm.output[1]
            )

    def check_make_manifest(self, url_root, url_list):
        """Check that make_manifest returns an appropriate DataFrame."""
        data = [5, 10, 20]
        config = zooniverseCutouts.ZooniverseCutoutsTask.ConfigClass()
        config.urlRoot = url_root
        cutouts = zooniverseCutouts.ZooniverseCutoutsTask(config=config)
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
        config = zooniverseCutouts.ZooniverseCutoutsTask.ConfigClass()
        config.size = 63
        cutouts = zooniverseCutouts.ZooniverseCutoutsTask(config=config, outputPath="something")
        other = pickle.loads(pickle.dumps(cutouts))
        self.assertEqual(cutouts.config.size, other.config.size)
        self.assertEqual(cutouts._outputPath, other._outputPath)


class TestZooniverseCutoutsMain(lsst.utils.tests.TestCase):
    """Test the commandline interface main() function via mocks."""

    def setUp(self):
        datadir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/")
        self.dbName = os.path.join(datadir, "apdb.sqlite3")
        self.repo = "/not/a/real/butler"
        self.collection = "mockRun"
        self.outputPath = "/an/output/path"
        self.configFile = os.path.join(datadir, "zooniverseCutoutsConfig.py")
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
            "zooniverseCutouts",
            f"--dbName={self.dbName}",
            f"--collections={self.collection}",
            f"-C={self.configFile}",
            f"--instrument={self.instrument}",
            self.repo,
            self.outputPath,
        ]
        with unittest.mock.patch.object(
            zooniverseCutouts.ZooniverseCutoutsTask, "run", autospec=True
        ) as run, unittest.mock.patch.object(sys, "argv", args):
            zooniverseCutouts.main()
            self.assertEqual(self._butler.call_args.args, (self.repo,))
            self.assertEqual(
                self._butler.call_args.kwargs, {"collections": [self.collection]}
            )
            # NOTE: can't easily test the `data` arg to run, as select_sources
            # reads in a random order every time.
            self.assertEqual(run.call_args.args[2], self._butler.return_value)
            self.assertEqual(run.call_args.args[3], self.outputPath)

    def test_main_args_no_collections(self):
        """Test with no collections argument."""
        args = [
            "zooniverseCutouts",
            f"--dbName={self.dbName}",
            f"-C={self.configFile}",
            f"--instrument={self.instrument}",
            self.repo,
            self.outputPath,
        ]
        with unittest.mock.patch.object(
            zooniverseCutouts.ZooniverseCutoutsTask, "run", autospec=True
        ) as run, unittest.mock.patch.object(sys, "argv", args):
            zooniverseCutouts.main()
            self.assertEqual(self._butler.call_args.args, (self.repo,))
            self.assertEqual(self._butler.call_args.kwargs, {"collections": None})
            self.assertEqual(run.call_args.args[2], self._butler.return_value)
            self.assertEqual(run.call_args.args[3], self.outputPath)

    def test_main_collection_list(self):
        """Test passing a list of collections."""
        collections = ["mock1", "mock2", "mock3"]
        args = [
            "zooniverseCutouts",
            f"--dbName={self.dbName}",
            f"--instrument={self.instrument}",
            self.repo,
            self.outputPath,
            f"-C={self.configFile}",
            "--collections",
        ]
        args.extend(collections)
        with unittest.mock.patch.object(
            zooniverseCutouts.ZooniverseCutoutsTask, "run", autospec=True
        ) as run, unittest.mock.patch.object(sys, "argv", args):
            zooniverseCutouts.main()
            self.assertEqual(self._butler.call_args.args, (self.repo,))
            self.assertEqual(
                self._butler.call_args.kwargs, {"collections": collections}
            )
            self.assertEqual(run.call_args.args[2], self._butler.return_value)
            self.assertEqual(run.call_args.args[3], self.outputPath)

    def test_main_args_limit_offset(self):
        """Test typical arguments to main()."""
        args = [
            "zooniverseCutouts",
            f"--dbName={self.dbName}",
            f"--collections={self.collection}",
            f"-C={self.configFile}",
            f"--instrument={self.instrument}",
            "--all",
            "--limit=5",
            self.repo,
            self.outputPath,
        ]
        with unittest.mock.patch.object(
            zooniverseCutouts.ZooniverseCutoutsTask, "write_images", autospec=True,
            return_value=[5]
        ) as write_images, unittest.mock.patch.object(
            zooniverseCutouts.ZooniverseCutoutsTask, "write_manifest", autospec=True
        ) as write_manifest, unittest.mock.patch.object(sys, "argv", args):
            zooniverseCutouts.main()
            self.assertEqual(self._butler.call_args.args, (self.repo,))
            self.assertEqual(
                self._butler.call_args.kwargs, {"collections": [self.collection]}
            )
            self.assertEqual(write_images.call_args.args[2], self._butler.return_value)
            self.assertEqual(write_images.call_args.args[3], self.outputPath)
            # The test apdb contains 15 sources, so we get the return of
            # `write_images` three times with `limit=5`
            self.assertEqual(write_manifest.call_args.args[1], [5, 5, 5])
            self.assertEqual(write_manifest.call_args.args[2], self.outputPath)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

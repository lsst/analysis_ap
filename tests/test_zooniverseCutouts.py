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
import pandas as pd
import PIL
import sys
import unittest
import tempfile

import lsst.afw.table
import lsst.geom
import lsst.meas.base.tests
from lsst.pex.config import FieldValidationError
import lsst.utils.tests

from lsst.analysis.ap import zooniverseCutouts


class TestZooniverseCutouts(lsst.utils.tests.TestCase):
    """Test that ZooniverseCutoutsTask generates images and manifest files
    correctly.
    """
    def setUp(self):
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(0, 0), lsst.geom.Point2I(100, 100))
        self.centroid = lsst.geom.Point2D(65, 70)
        dataset = lsst.meas.base.tests.TestDataset(bbox)
        dataset.addSource(instFlux=1e5, centroid=self.centroid)
        self.science, self.scienceCat = dataset.realize(noise=1000.0, schema=dataset.makeMinimalSchema())
        lsst.afw.table.updateSourceCoords(self.science.wcs, self.scienceCat)
        self.skyCenter = self.scienceCat[0].getCoord()
        self.template, self.templateCat = dataset.realize(noise=5.0, schema=dataset.makeMinimalSchema())
        # A simple image difference to have something to plot.
        self.difference = lsst.afw.image.ExposureF(self.science, deep=True)
        self.difference.image -= self.template.image

    def test_generate_image(self):
        """Test that we get some kind of image out.

        It's useful to have a person look at the output via:
            im.show()
        """
        cutouts = zooniverseCutouts.ZooniverseCutoutsTask()
        cutout = cutouts.generate_image(self.science, self.template, self.difference,
                                        self.skyCenter)
        with PIL.Image.open(cutout) as im:
            # NOTE: uncomment this to show the resulting image.
            # im.show()
            # NOTE: the dimensions here are determined by the matplotlib figure
            # size (in inches) and the dpi (default=100), plus borders.
            self.assertEqual(im.height, 233)
            self.assertEqual(im.width, 630)

    def test_generate_image_larger_cutout(self):
        """A different cutout size: the resulting cutout image is the same
        size but shows more pixels.
        """
        config = zooniverseCutouts.ZooniverseCutoutsTask.ConfigClass()
        config.size = 100
        cutouts = zooniverseCutouts.ZooniverseCutoutsTask(config=config)
        cutout = cutouts.generate_image(self.science, self.template, self.difference,
                                        self.skyCenter)
        with PIL.Image.open(cutout) as im:
            # NOTE: uncomment this to show the resulting image.
            # im.show()
            # NOTE: the dimensions here are determined by the matplotlib figure
            # size (in inches) and the dpi (default=100), plus borders.
            self.assertEqual(im.height, 233)
            self.assertEqual(im.width, 630)

    def test_write_images(self):
        """Test that images get written to a temporary directory."""
        data = pd.DataFrame(data={"diaSourceId": [5, 10],
                                  "ra": [45.001, 45.002],
                                  "decl": [45.0, 45.001],
                                  "detector": [50, 60],
                                  "visit": [1234, 5678],
                                  "instrument": ["mockCam", "mockCam"]})
        butler = unittest.mock.Mock(spec=lsst.daf.butler.Butler)
        # we don't care what the output images look like here, just that
        # butler.get() returns an Exposure for every call.
        butler.get.return_value = self.science

        with tempfile.TemporaryDirectory() as path:
            config = zooniverseCutouts.ZooniverseCutoutsTask.ConfigClass()
            cutouts = zooniverseCutouts.ZooniverseCutoutsTask(config=config)
            result = cutouts.write_images(data, butler, path)
            self.assertEqual(result, list(data['diaSourceId']))
            for file in ("images/5.png", "images/10.png"):
                filename = os.path.join(path, file)
                self.assertTrue(os.path.exists(filename))
                with PIL.Image.open(filename) as image:
                    self.assertEqual(image.format, "PNG")

    def test_write_images_exception(self):
        """Test that write_images() catches errors in loading data.
        """
        data = pd.DataFrame(data={"diaSourceId": [5, 10],
                                  "ra": [45.001, 45.002],
                                  "decl": [45.0, 45.001],
                                  "detector": [50, 60],
                                  "visit": [1234, 5678],
                                  "instrument": ["mockCam", "mockCam"]})
        butler = unittest.mock.Mock(spec=lsst.daf.butler.Butler)
        err = "Dataset not found"
        butler.get.side_effect = LookupError(err)

        with tempfile.TemporaryDirectory() as path:
            config = zooniverseCutouts.ZooniverseCutoutsTask.ConfigClass()
            cutouts = zooniverseCutouts.ZooniverseCutoutsTask(config=config)

            with self.assertLogs("zooniverseCutouts", "ERROR") as cm:
                cutouts.write_images(data, butler, path)
            self.assertIn("LookupError processing diaSourceId 5: Dataset not found", cm.output[0])
            self.assertIn("LookupError processing diaSourceId 10: Dataset not found", cm.output[1])

    def check_make_manifest(self, url_root, url_list):
        """Check that make_manifest returns an appropriate DataFrame.
        """
        data = [5, 10, 20]
        config = zooniverseCutouts.ZooniverseCutoutsTask.ConfigClass()
        config.urlRoot = url_root
        cutouts = zooniverseCutouts.ZooniverseCutoutsTask(config=config)
        manifest = cutouts.make_manifest(data)
        self.assertEqual(manifest['metadata:diaSourceId'].to_list(),
                         [5, 10, 20])
        self.assertEqual(manifest['location:1'].to_list(), url_list)

    def test_make_manifest(self):
        # check without an ending slash
        root = "http://example.org/zooniverse"
        url_list = [f"{root}/images/5.png",
                    f"{root}/images/10.png",
                    f"{root}/images/20.png"]
        self.check_make_manifest(root, url_list)

        # check with an ending slash
        root = "http://example.org/zooniverse/"
        url_list = [f"{root}images/5.png",
                    f"{root}images/10.png",
                    f"{root}images/20.png"]
        self.check_make_manifest(root, url_list)


class TestZooniverseCutoutsMain(lsst.utils.tests.TestCase):
    """Test the commandline interface main() function via mocks.
    """
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
        self._butler.return_value.registry.dimensions.makePacker.return_value.unpack = mock_unpacker
        self.addCleanup(butlerPatch.stop)

    def test_main_args(self):
        """Test typical arguments to main().
        """
        args = ["zooniverseCutouts",
                f"--dbName={self.dbName}",
                f"--collections={self.collection}",
                f"-C={self.configFile}",
                f"--instrument={self.instrument}",
                self.repo,
                self.outputPath
                ]
        with unittest.mock.patch.object(zooniverseCutouts.ZooniverseCutoutsTask, "run",
                                        autospec=True) as run, \
                unittest.mock.patch.object(sys, "argv", args):
            zooniverseCutouts.main()
            self.assertEqual(self._butler.call_args.args, (self.repo,))
            self.assertEqual(self._butler.call_args.kwargs, {"collections": [self.collection]})
            # NOTE: can't easily test the `data` arg to run, as select_sources
            # reads in a random order every time.
            self.assertEqual(run.call_args.args[2], self._butler.return_value)
            self.assertEqual(run.call_args.args[3], self.outputPath)

    def test_main_args_no_config_fails(self):
        """Test that not passing a config file fails because urlRoot is None.
        """
        args = ["zooniverseCutouts",
                f"--dbName={self.dbName}",
                f"--collections={self.collection}",
                f"--instrument={self.instrument}",
                self.repo,
                self.outputPath
                ]
        with unittest.mock.patch.object(sys, "argv", args), \
                self.assertRaisesRegex(FieldValidationError, "Field 'urlRoot' failed validation"):
            zooniverseCutouts.main()

    def test_main_args_no_collections(self):
        """Test with no collections argument.
        """
        args = ["zooniverseCutouts",
                f"--dbName={self.dbName}",
                f"-C={self.configFile}",
                f"--instrument={self.instrument}",
                self.repo,
                self.outputPath
                ]
        with unittest.mock.patch.object(zooniverseCutouts.ZooniverseCutoutsTask, "run",
                                        autospec=True) as run, \
                unittest.mock.patch.object(sys, "argv", args):
            zooniverseCutouts.main()
            self.assertEqual(self._butler.call_args.args, (self.repo,))
            self.assertEqual(self._butler.call_args.kwargs, {"collections": None})
            self.assertEqual(run.call_args.args[2], self._butler.return_value)
            self.assertEqual(run.call_args.args[3], self.outputPath)

    def test_main_collection_list(self):
        """Test passing a list of collections.
        """
        collections = ["mock1", "mock2", "mock3"]
        args = ["zooniverseCutouts",
                f"--dbName={self.dbName}",
                f"--instrument={self.instrument}",
                self.repo,
                self.outputPath,
                f"-C={self.configFile}",
                "--collections"]
        args.extend(collections)
        with unittest.mock.patch.object(zooniverseCutouts.ZooniverseCutoutsTask, "run",
                                        autospec=True) as run, \
                unittest.mock.patch.object(sys, "argv", args):
            zooniverseCutouts.main()
            self.assertEqual(self._butler.call_args.args, (self.repo,))
            self.assertEqual(self._butler.call_args.kwargs, {"collections": collections})
            self.assertEqual(run.call_args.args[2], self._butler.return_value)
            self.assertEqual(run.call_args.args[3], self.outputPath)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

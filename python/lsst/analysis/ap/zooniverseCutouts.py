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

"""Construct template/image/difference cutouts for upload to Zooniverse.
"""

__all__ = ["ZooniverseCutoutsConfig", "ZooniverseCutoutsTask"]

import argparse
import functools
import io
import logging
import os
import pathlib
import pandas as pd

import lsst.dax.apdb
import lsst.pex.config as pexConfig
import lsst.pipe.base

from . import apdbUtils


class ZooniverseCutoutsConfig(pexConfig.Config):
    size = pexConfig.Field(
        doc="Width of cutout to extract for image from science, template, and difference exposures.",
        dtype=int,
        default=30
    )
    urlRoot = pexConfig.Field(
        doc="URL that the resulting images will be served to Zooniverse from, for the manifest file.",
        dtype=str,
        optional=False
    )
    diffImageType = pexConfig.Field(
        doc="Dataset type of template and difference image to use for cutouts; "
            "Will have '_warpedExp' and '_differenceExp' appended for butler.get(), respectively.",
        dtype=str,
        default="deepDiff"
    )


class ZooniverseCutoutsTask(lsst.pipe.base.Task):
    """Generate cutouts and a manifest for upload to a Zooniverse project.
    """
    ConfigClass = ZooniverseCutoutsConfig
    _DefaultName = "zooniverseCutouts"

    def run(self, data, butler, outputPath):
        """Generate cutouts images and a manifest for upload to Zooniverse
        from a collection of sources.

        Parameters
        ----------
        data : `pandas.DataFrame`
            The DiaSources to extract cutouts for. Must contain at least these
            fields: ``ra, dec, diaSourceId, detector, visit, instrument``.
        butler : `lsst.daf.butler.Butler`
            The butler connection to use to load the data; create it with the
            collections you wish to load images from.
        outputPath : `str`
            The path to write the output to; manifest goes here, while the
            images themselves go into ``outputPath/images/``.
        """
        result = self.write_images(data, butler, outputPath)
        manifest = self.make_manifest(result)
        manifest.to_csv(os.path.join(outputPath, "manifest.csv"), index=False)
        self.log.info("Wrote %d images to %s", len(result), outputPath)

    @staticmethod
    def _make_path(id, base_path):
        """Return a URL or file path for this source.

        Parameters
        ----------
        id : `int`
            The source id of the source.
        base_path : `str`
            Base URL or directory path, with no ending ``/``.

        Returns
        -------
        path : `str`
            Formatted URL or path.
        """
        return f"{base_path}/images/{id}.png"

    def make_manifest(self, sources):
        """Return a Zooniverse manifest attaching image URLs to source ids.

        Parameters
        ----------
        sources : `list` [`int`]
            The diaSourceIds of the sources that had cutouts succesfully made.

        Returns
        -------
        manifest : `pandas.DataFrame`
            The formatted URL manifest for upload to Zooniverse.
        """
        manifest = pd.DataFrame()
        manifest['external_id'] = sources
        manifest['location:1'] = [self._make_path(x, self.config.urlRoot.rstrip('/')) for x in sources]
        manifest['metadata:diaSourceId'] = sources
        return manifest

    def write_images(self, data, butler, outputPath):
        """Make the 3-part cutout images for each requested source and write
        them to disk.

        Creates a ``images/`` subdirectory in ``outputPath`` if one
        does not already exist; images are written there as PNG files.

        Parameters
        ----------
        data : `pandas.DataFrame`
            The DiaSources to extract cutouts for. Must contain at least these
            fields: ``ra, dec, diaSourceId, detector, visit, instrument``.
        butler : `lsst.daf.butler.Butler`
            The butler connection to use to load the data; create it with the
            collections you wish to load images from.
        outputPath : `str`
            The path to write the output to; manifest goes here, while the
            images themselves go into ``outputPath/images/``.
        """
        @functools.lru_cache(maxsize=16)
        def get_exposures(instrument, detector, visit):
            """Return science, template, difference exposures, and use a small
            cache so we don't have to re-read files as often.

            NOTE: Closure because it needs access to the local-scoped butler,
            as lru_cache can't have mutable args in the decorated method.

            If we redo this all to work with BPS or other parallelized
            systems, or get good butler-side caching, we could remove the
            lru_cache above.
            """
            dataId = {'instrument': instrument, 'detector': detector, 'visit': visit}
            return (butler.get('calexp', dataId),
                    butler.get(f'{self.config.diffImageType}_warpedExp', dataId),
                    butler.get(f'{self.config.diffImageType}_differenceExp', dataId))

        # Create a subdirectory for the images.
        pathlib.Path(os.path.join(outputPath, "images")).mkdir(exist_ok=True)

        result = []
        for index, source in data.iterrows():
            try:
                center = lsst.geom.SpherePoint(source['ra'], source['decl'], lsst.geom.degrees)
                science, template, difference = get_exposures(source['instrument'],
                                                              source['detector'],
                                                              source['visit'])
                image = self.generate_image(science, template, difference, center)
                with open(self._make_path(source.loc['diaSourceId'], outputPath), "wb") as outfile:
                    outfile.write(image.getbuffer())
                result.append(source.loc['diaSourceId'])
            except LookupError as e:
                self.log.error(f"{e.__class__.__name__} processing diaSourceId {source['diaSourceId']}: {e}")
        return result

    def generate_image(self, science, template, difference, center):
        """Get a 3-part cutout image to save to disk, for a single source.

        Parameters
        ----------
        science : `lsst.afw.image.ExposureF`
            Science exposure to include in the cutout.
        template : `lsst.afw.image.ExposureF`
            Matched template exposure to include in the cutout.
        difference : `lsst.afw.image.ExposureF`
             Matched science minus template exposure to include in the cutout.
        center : `lsst.geom.SpherePoint`
            Center of the source to be cut out of each image.

        Returns
        -------
        image: `io.BytesIO`
            The generated image, to be output to a file or displayed on screen.
        """
        size = lsst.geom.Extent2I(self.config.size, self.config.size)
        return self._plot_cutout(science.getCutout(center, size),
                                 template.getCutout(center, size),
                                 difference.getCutout(center, size)
                                 )

    def _plot_cutout(self, science, template, difference):
        """Plot the cutouts for a source in one image.

        Parameters
        ----------
        science : `lsst.afw.image.ExposureF`
            Cutout Science exposure to include in the image.
        template : `lsst.afw.image.ExposureF`
            Cutout template exposure to include in the image.
        difference : `lsst.afw.image.ExposureF`
             Cutout science minus template exposure to include in the image.

        Returns
        -------
        image: `io.BytesIO`
            The generated image, to be output to a file via
            `image.write(filename)` or displayed on screen.
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm
        import astropy.visualization as aviz

        # TODO DM-32014: how do we color masked pixels (including edges)?

        def plot_one_image(ax, data, name):
            """Plot a normalized image on an axis."""
            if name == 'Difference':
                norm = aviz.ImageNormalize(data, interval=aviz.ZScaleInterval(),
                                           stretch=aviz.LinearStretch())
            else:
                norm = aviz.ImageNormalize(data, interval=aviz.MinMaxInterval(),
                                           stretch=aviz.AsinhStretch(a=0.1))
            ax.imshow(data, cmap=cm.bone, interpolation="none", norm=norm)
            ax.axis('off')
            ax.set_title(name)

        try:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            plot_one_image(ax1, template.image.array, "Template")
            plot_one_image(ax2, science.image.array, "Science")
            plot_one_image(ax3, difference.image.array, "Difference")
            plt.tight_layout()

            output = io.BytesIO()
            plt.savefig(output, bbox_inches="tight", format="png")
            output.seek(0)  # to ensure opening the image starts from the front
        finally:
            plt.close(fig)

        return output


def build_argparser():
    """Construct an argument parser for the ``zooniverseCutouts`` script.

    Returns
    -------
    argparser : `argparse.ArgumentParser`
        The argument parser that defines the ``zooniverseCutouts``
        command-line interface.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='More information is available at https://pipelines.lsst.io.'
    )

    apdbArgs = parser.add_argument_group("apdb connection")
    apdbArgs.add_argument("--dbName", required=True,
                          help="Full path (sqlite) or name on lsst-pg-devl (postgres) of the APDB to load.")
    apdbArgs.add_argument("--dbType", default="sqlite", choices=["sqlite", "postgres"],
                          help="Type of database to connect to.")
    apdbArgs.add_argument("--schema",
                          help="Schema to connect to; only for 'dbName=postgres'.")

    parser.add_argument("-n", default=1000, type=int,
                        help="Number of sources to load randomly from the APDB.")

    parser.add_argument("--instrument", required=True,
                        help="Instrument short-name (e.g. 'DECam') of the data being loaded.")
    parser.add_argument("-C", "--configFile",
                        help="File containing the ZooniverseCutoutsConfig to load.")
    parser.add_argument("--collections", nargs="*",
                        help=("Butler collection(s) to load data from."
                              " If not specified, will search all butler collections, "
                              "which may be very slow."))
    parser.add_argument("repo",
                        help="Path to Butler repository to load data from.")
    parser.add_argument("outputPath",
                        help="Path to write the output images and manifest to; "
                        "manifest is written here, while the images go to `OUTPUTPATH/images/`.")
    return parser


def select_sources(dbName, dbType, schema, butler, instrument, n):
    """Load an APDB and select n objects randomly from it.

    Parameters
    ----------
    dbName : `str`
        If dbType is sqlite, *full path* to the APDB on lsst-dev.
        If dbType is postgres, name of the APDB on lsst-pg-devel1.
    dbType : `str`
        Either 'sqlite' or 'postgres'.
    schema : `str`
        Required if dbType is postgres, ignored for sqlite.
    butler : `lsst.daf.butler.Butler`
        A butler instance to use to fill out detector/visit information.
    instrument : `str`
        Instrument that produced these data, to fill out a new column.
    n : `int`
        Number of sources to randomly select from the APDB.

    Returns
    -------
    sources : `pandas.DataFrame`
        The loaded DiaSource data.
    """
    connection = apdbUtils.connectToApdb(dbName, dbType, schema)
    sources = pd.read_sql_query(f'select * from "DiaSource" ORDER BY RANDOM() LIMIT {n};', connection)
    apdbUtils.addTableMetadata(sources, instrument=instrument, butler=butler)
    return sources


def run_cutouts(args):
    """Run ZooniverseCutoutsTask on the parsed commandline arguments.

    Parameters
    ----------
    args : `argparse.Namespace`
        The parsed commandline arguments.
    """
    # We have to initialize the logger manually on the commandline.
    logging.basicConfig(level=logging.INFO, format="{name} {levelname}: {message}", style="{")

    butler = lsst.daf.butler.Butler(args.repo, collections=args.collections)
    data = select_sources(args.dbName, args.dbType, args.schema, butler, args.instrument, args.n)

    config = ZooniverseCutoutsConfig()
    if args.configFile is not None:
        config.load(os.path.expanduser(args.configFile))
    config.validate()
    config.freeze()
    cutouts = ZooniverseCutoutsTask(config=config)
    cutouts.run(data, butler, args.outputPath)


def main():
    args = build_argparser().parse_args()
    run_cutouts(args)

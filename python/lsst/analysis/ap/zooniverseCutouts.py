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

"""Construct template/image/difference cutouts for upload to Zooniverse, or
to just to view as images.
"""

__all__ = ["ZooniverseCutoutsConfig", "ZooniverseCutoutsTask"]

import argparse
import functools
import io
import itertools
import logging
import multiprocessing
import os
import pathlib

import astropy.units as u
import numpy as np
import pandas as pd

from lsst.ap.association import UnpackApdbFlags
import lsst.dax.apdb
import lsst.pex.config as pexConfig
import lsst.pipe.base
import lsst.utils

from . import legacyApdbUtils


class ZooniverseCutoutsConfig(pexConfig.Config):
    size = pexConfig.Field(
        doc="Width of cutout to extract for image from science, template, and difference exposures.",
        dtype=int,
        default=30,
    )
    urlRoot = pexConfig.Field(
        doc="URL that the resulting images will be served to Zooniverse from, for the manifest file. "
            "If not set, no manifest file will be written.",
        dtype=str,
        default=None,
        optional=True,
    )
    diffImageType = pexConfig.Field(
        doc="Dataset type of template and difference image to use for cutouts; "
            "Will have '_templateExp' and '_differenceExp' appended for butler.get(), respectively.",
        dtype=str,
        default="deepDiff",
    )
    addMetadata = pexConfig.Field(
        doc="Annotate the cutouts with catalog metadata, including coordinates, fluxes, flags, etc.",
        dtype=bool,
        default=False
    )
    n_processes = pexConfig.Field(
        doc="Number of processes to use when making cutout images."
            " 0 means do not use multiprocessing.",
        dtype=int,
        default=0
    )
    chunk_size = pexConfig.Field(
        doc="Chunk up files into subdirectories, with at most this many files per directory."
            " None (default) means write all the files to one `images/` directory.",
        dtype=int,
        default=None,
        optional=True
    )


class ZooniverseCutoutsTask(lsst.pipe.base.Task):
    """Generate cutouts and a manifest for upload to a Zooniverse project.

    Parameters
    ----------
    outputPath : `str`
        The path to write the output to; manifest goes here, while the
        images themselves go into ``outputPath/images/``.
    """
    ConfigClass = ZooniverseCutoutsConfig
    _DefaultName = "zooniverseCutouts"

    def __init__(self, *, outputPath, **kwargs):
        super().__init__(**kwargs)
        self._outputPath = outputPath
        self.path_manager = PathManager(outputPath, chunk_size=self.config.chunk_size)

    def _reduce_kwargs(self):
        # to allow pickling of this Task
        kwargs = super()._reduce_kwargs()
        kwargs["outputPath"] = self._outputPath
        return kwargs

    def run(self, data, butler):
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

        Returns
        -------
        source_ids : `list`
            DiaSourceIds of cutout images that were generated.
        """
        result = self.write_images(data, butler)
        self.write_manifest(result)
        self.log.info("Wrote %d images to %s", len(result), self.path_manager(filename=""))
        return result

    def write_manifest(self, sources):
        """Save a Zooniverse manifest attaching image URLs to source ids.

        Parameters
        ----------
        sources : `list` [`int`]
            The diaSourceIds of the sources that had cutouts succesfully made.
        """
        if self.config.urlRoot is not None:
            manifest = self.make_manifest(sources)
            manifest.to_csv(self.path_manager(fileame="manifest.csv"), index=False)
        else:
            self.log.warning("No urlRoot provided, so no manifest file written.")

    def _make_manifest(self, sources):
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
        path_manager = PathManager(self.config.urlRoot)
        manifest = pd.DataFrame()
        manifest["external_id"] = sources
        manifest["location:1"] = [path_manager(x) for x in sources]
        manifest["metadata:diaSourceId"] = sources
        return manifest

    def write_images(self, data, butler):
        """Make the 3-part cutout images for each requested source and write
        them to disk.

        Creates a ``images/`` subdirectory via path_manager if one
        does not already exist; images are written there as PNG files.

        Parameters
        ----------
        data : `pandas.DataFrame`
            The DiaSources to extract cutouts for. Must contain at least these
            fields: ``ra, dec, diaSourceId, detector, visit, instrument``.
        butler : `lsst.daf.butler.Butler`
            The butler connection to use to load the data; create it with the
            collections you wish to load images from.

        Returns
        -------
        sources : `list`
            DiaSourceIds that had cutouts made.
        """
        # Ignore divide-by-zero and log-of-negative-value messages.
        seterr_dict = np.seterr(divide="ignore", invalid="ignore")
        flag_map = os.path.join(lsst.utils.getPackageDir("ap_association"), "data/association-flag-map.yaml")
        unpacker = UnpackApdbFlags(flag_map, "DiaSource")
        flags = unpacker.unpack(data["flags"], "flags")

        # Create a subdirectory for the images.
        pathlib.Path(self.path_manager(filename="images")).mkdir(exist_ok=True)

        sources = []
        if self.config.n_processes > 0:
            with multiprocessing.Pool(self.config.n_processes) as pool:
                sources = pool.starmap(self._do_one_source, zip(data.to_records(), flags,
                                                                itertools.repeat(butler)))
        else:
            for i, source in enumerate(data.to_records()):
                temp = self._do_one_source(source, flags[i], butler)
                if temp is not None:
                    sources.append(temp)

        np.seterr(**seterr_dict)
        return sources

    def _do_one_source(self, source, flags, butler):
        """Make cutouts for one diaSource.
        """
        @functools.lru_cache(maxsize=4)
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
                    butler.get(f'{self.config.diffImageType}_templateExp', dataId),
                    butler.get(f'{self.config.diffImageType}_differenceExp', dataId))

        try:
            center = lsst.geom.SpherePoint(source["ra"], source["decl"], lsst.geom.degrees)
            science, template, difference = get_exposures(source["instrument"], source["detector"], source["visit"])
            image = self.generate_image(science, template, difference, center,
                                        source=source if self.config.addMetadata else None,
                                        flags=flags if self.config.addMetadata else None)
            self.path_manager.create_path(source["diaSourceId"])
            with open(self.path_manager(source["diaSourceId"]), "wb") as outfile:
                outfile.write(image.getbuffer())
            return source["diaSourceId"]
        except LookupError as e:
            self.log.error(
                f"{e.__class__.__name__} processing diaSourceId {source['diaSourceId']}: {e}"
            )
            return None
        except Exception as e:
            # ensure other exceptions are interpretable when multiprocessing
            import traceback
            traceback.print_exc()
            raise e

    def generate_image(self, science, template, difference, center, source=None, flags=None):
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
        source : `numpy.record`, optional
            DiaSource record for this cutout, to add metadata to the image.
        flags : `str`, optional
            Unpacked bits from the ``flags`` field in ``source``.
            Required if ``source`` is not None.

        Returns
        -------
        image: `io.BytesIO`
            The generated image, to be output to a file or displayed on screen.
        """
        if (source is None) ^ (flags is None):
            raise RuntimeError("Must pass both `source` and `flags` together.")
        size = lsst.geom.Extent2I(self.config.size, self.config.size)
        return self._plot_cutout(
            science.getCutout(center, size),
            template.getCutout(center, size),
            difference.getCutout(center, size),
            source=source,
            flags=flags
        )

    def _plot_cutout(self, science, template, difference, source=None, flags=None):
        """Plot the cutouts for a source in one image.

        Parameters
        ----------
        science : `lsst.afw.image.ExposureF`
            Cutout Science exposure to include in the image.
        template : `lsst.afw.image.ExposureF`
            Cutout template exposure to include in the image.
        difference : `lsst.afw.image.ExposureF`
             Cutout science minus template exposure to include in the image.
        source : `numpy.record`, optional
            DiaSource record for this cutout, to add metadata to the image.
        flags : `str`, optional
            Unpacked bits from the ``flags`` field in ``source``.

        Returns
        -------
        image: `io.BytesIO`
            The generated image, to be output to a file via
            `image.write(filename)` or displayed on screen.
        """
        import astropy.visualization as aviz
        import matplotlib
        matplotlib.use("AGG")
        # Force matplotlib defaults
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        import matplotlib.pyplot as plt
        from matplotlib import cm

        # TODO DM-32014: how do we color masked pixels (including edges)?

        def plot_one_image(ax, data, name):
            """Plot a normalized image on an axis."""
            if name == "Difference":
                norm = aviz.ImageNormalize(
                    data, interval=aviz.ZScaleInterval(), stretch=aviz.LinearStretch()
                )
            else:
                norm = aviz.ImageNormalize(
                    data,
                    interval=aviz.MinMaxInterval(),
                    stretch=aviz.AsinhStretch(a=0.1),
                )
            ax.imshow(data, cmap=cm.bone, interpolation="none", norm=norm)
            ax.axis("off")
            ax.set_title(name)

        try:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, constrained_layout=True)
            plot_one_image(ax1, template.image.array, "Template")
            plot_one_image(ax2, science.image.array, "Science")
            plot_one_image(ax3, difference.image.array, "Difference")
            plt.tight_layout()
            if source is not None:
                _annotate_image(fig, source, flags)

            output = io.BytesIO()
            plt.savefig(output, bbox_inches="tight", format="png")
            output.seek(0)  # to ensure opening the image starts from the front
        finally:
            plt.close(fig)

        return output


def _annotate_image(fig, source, flags):
    """Annotate the cutouts image with metadata and flags.

    Parameters
    ----------
    fig : `matplotlib.Figure`
        Figure to be annotated.
    source : `numpy.record`
        DiaSource record of the object being plotted.
    flags : `str`, optional
        Unpacked bits from the ``flags`` field in ``source``.
    """
    # Names of flags fields to add a flag label to the image, using any().
    flags_psf = ["slot_PsfFlux_flag", "slot_PsfFlux_flag_noGoodPixels", "slot_PsfFlux_flag_edge"]
    flags_aperture = ["slot_ApFlux_flag", "slot_ApFlux_flag_apertureTruncated"]
    flags_forced = ["ip_diffim_forced_PsfFlux_flag", "ip_diffim_forced_PsfFlux_flag_noGoodPixels",
                    "ip_diffim_forced_PsfFlux_flag_edge"]
    flags_edge = ["base_PixelFlags_flag_edge"]
    flags_interp = ["base_PixelFlags_flag_interpolated", "base_PixelFlags_flag_interpolatedCenter"]
    flags_saturated = ["base_PixelFlags_flag_saturated", "base_PixelFlags_flag_saturatedCenter"]
    flags_cr = ["base_PixelFlags_flag_cr", "base_PixelFlags_flag_crCenter"]
    flags_bad = ["base_PixelFlags_flag_bad"]
    flags_suspect = ["base_PixelFlags_flag_suspect", "base_PixelFlags_flag_suspectCenter"]
    flags_centroid = ["slot_Centroid_flag"]
    flags_centroid_pos = ["slot_Centroid_pos_flag"]
    flags_centroid_neg = ["slot_Centroid_neg_flag"]
    flags_shape = ["slot_Shape_flag", "slot_Shape_flag_unweightedBad", "slot_Shape_flag_unweighted",
                   "slot_Shape_flag_shift", "slot_Shape_flag_maxIter", "slot_Shape_flag_psf"]

    flag_color = "red"
    text_color = "grey"
    # NOTE: fig.text coordinates are in fractions of the figure.
    fig.text(0, 0.95, "diaSourceId:", color=text_color)
    fig.text(0.145, 0.95, f"{source['diaSourceId']}")
    fig.text(0.43, 0.95, f"{source['instrument']}", fontweight="bold")
    fig.text(0.64, 0.95, "detector:", color=text_color)
    fig.text(0.74, 0.95, f"{source['detector']}")
    fig.text(0.795, 0.95, "visit:", color=text_color)
    fig.text(0.85, 0.95, f"{source['visit']}")
    fig.text(0.95, 0.95, f"{source['filterName']}")

    fig.text(0.0, 0.91, "ra:", color=text_color)
    fig.text(0.037, 0.91, f"{source['ra']:.8f}")
    fig.text(0.21, 0.91, "dec:", color=text_color)
    fig.text(0.265, 0.91, f"{source['decl']:+.8f}")
    fig.text(0.50, 0.91, "detection S/N:", color=text_color)
    fig.text(0.66, 0.91, f"{source['snr']:6.1f}")
    fig.text(0.75, 0.91, "PSF chi2:", color=text_color)
    fig.text(0.85, 0.91, f"{source['psChi2']/source['psNdata']:6.2f}")

    fig.text(0.0, 0.87, "PSF (nJy):", color=flag_color if any(flags[flags_psf]) else text_color)
    fig.text(0.25, 0.87, f"{source['psFlux']:8.1f}", horizontalalignment='right')
    fig.text(0.252, 0.87, "+/-", color=text_color)
    fig.text(0.29, 0.87, f"{source['psFluxErr']:8.1f}")
    fig.text(0.40, 0.87, "S/N:", color=text_color)
    fig.text(0.45, 0.87, f"{abs(source['psFlux']/source['psFluxErr']):6.2f}")

    # NOTE: yellow is hard to read on white; use goldenrod instead.
    if any(flags[flags_edge]):
        fig.text(0.55, 0.87, "EDGE", color="goldenrod", fontweight="bold")
    if any(flags[flags_interp]):
        fig.text(0.62, 0.87, "INTERP", color="green", fontweight="bold")
    if any(flags[flags_saturated]):
        fig.text(0.72, 0.87, "SAT", color="green", fontweight="bold")
    if any(flags[flags_cr]):
        fig.text(0.77, 0.87, "CR", color="magenta", fontweight="bold")
    if any(flags[flags_bad]):
        fig.text(0.81, 0.87, "BAD", color="red", fontweight="bold")
    if source['isDipole']:
        fig.text(0.87, 0.87, "DIPOLE", color="indigo", fontweight="bold")

    fig.text(0.0, 0.83, "ap (nJy):", color=flag_color if any(flags[flags_aperture]) else text_color)
    fig.text(0.25, 0.83, f"{source['apFlux']:8.1f}", horizontalalignment='right')
    fig.text(0.252, 0.83, "+/-", color=text_color)
    fig.text(0.29, 0.83, f"{source['apFluxErr']:8.1f}")
    fig.text(0.40, 0.83, "S/N:", color=text_color)
    fig.text(0.45, 0.83, f"{abs(source['apFlux']/source['apFluxErr']):#6.2f}")

    if any(flags[flags_suspect]):
        fig.text(0.55, 0.83, "SUS", color="goldenrod", fontweight="bold")
    if any(flags[flags_centroid]):
        fig.text(0.60, 0.83, "CENTROID", color="red", fontweight="bold")
    if any(flags[flags_centroid_pos]):
        fig.text(0.73, 0.83, "CEN+", color="chocolate", fontweight="bold")
    if any(flags[flags_centroid_neg]):
        fig.text(0.80, 0.83, "CEN-", color="blue", fontweight="bold")
    if any(flags[flags_shape]):
        fig.text(0.87, 0.83, "SHAPE", color="red", fontweight="bold")

    fig.text(0.0, 0.79, "total (nJy):", color=flag_color if any(flags[flags_forced]) else text_color)
    fig.text(0.25, 0.79, f"{source['totFlux']:8.1f}", horizontalalignment='right')
    fig.text(0.252, 0.79, "+/-", color=text_color)
    fig.text(0.29, 0.79, f"{source['totFluxErr']:8.1f}")
    fig.text(0.40, 0.79, "S/N:", color=text_color)
    fig.text(0.45, 0.79, f"{abs(source['totFlux']/source['totFluxErr']):6.2f}")
    fig.text(0.55, 0.79, "ABmag:", color=text_color)
    fig.text(0.635, 0.79, f"{(source['totFlux']*u.nanojansky).to_value(u.ABmag):.3f}")


class PathManager:
    """Manage paths to local files, chunked directories, and s3 buckets.

    Parameters
    ----------
    root : `str`
        Root file path to manage.
    chunk_size : `int`, optional
        How many files per directory?
    """
    def __init__(self, root, chunk_size=None):
        self._root = root
        if chunk_size is not None and chunk_size % 10 != 0:
            raise RuntimeError(f"PathManager file chunking must be a multiple of 10, got {chunk_size}.")
        self._chunk_size = chunk_size

    def __call__(self, id=None, filename=None):
        """Return the full path to this diaSourceId cutout.

        Parameters
        ----------
        id : `int`
            Description
        filename : None, optional
            Description

        Returns
        -------
        TYPE
            Description
        """
        def chunker(id, size):
            return (id // size)*size

        if id is not None:
            if self._chunk_size is not None:
                return os.path.join(self._root, f"images/{chunker(id, self._chunk_size)}/{id}.png")
            else:
                return os.path.join(self._root, f"images/{id}.png")
        elif filename is not None:
            return os.path.join(self._root, filename)

    def create_path(self, id=None, filename=None):
        """Summary

        Parameters
        ----------
        id : None, optional
            Description
        filename : None, optional
            Description
        """
        path = os.path.dirname(self(id=id, filename=filename))
        os.makedirs(path, exist_ok=True)


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
        epilog="More information is available at https://pipelines.lsst.io.",
    )

    apdbArgs = parser.add_argument_group("apdb connection")
    apdbArgs.add_argument(
        "--dbName",
        required=True,
        help="Full path (sqlite) or name on lsst-pg-devl (postgres) of the APDB to load.",
    )
    apdbArgs.add_argument(
        "--dbType",
        default="sqlite",
        choices=["sqlite", "postgres"],
        help="Type of database to connect to (default='sqlite').",
    )
    apdbArgs.add_argument(
        "--schema", help="Schema to connect to; only for 'dbType=postgres'."
    )

    parser.add_argument(
        "--limit",
        default=5,
        type=int,
        help="Number of sources to load from the APDB (default=5), or the "
             "number of sources to load per 'page' when `--all` is set.",
    )
    parser.add_argument(
        "--all",
        default=False,
        action="store_true",
        help="Process all the sources; --limit then becomes the 'page size' to chunk the DB into.",
    )

    parser.add_argument(
        "--instrument",
        required=True,
        help="Instrument short-name (e.g. 'DECam') of the data being loaded.",
    )
    parser.add_argument(
        "-C",
        "--configFile",
        help="File containing the ZooniverseCutoutsConfig to load.",
    )
    parser.add_argument(
        "--collections",
        nargs="*",
        help=(
            "Butler collection(s) to load data from."
            " If not specified, will search all butler collections, "
            "which may be very slow."
        ),
    )
    parser.add_argument("repo", help="Path to Butler repository to load data from.")
    parser.add_argument(
        "outputPath",
        help="Path to write the output images and manifest to; "
        "manifest is written here, while the images go to `OUTPUTPATH/images/`.",
    )
    return parser


def select_sources(dbName, dbType, schema, butler, instrument, limit):
    """Load an APDB and select n objects from it.

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
    limit : `int`
        Number of sources to select from the APDB.

    Returns
    -------
    sources : `pandas.DataFrame`
        The loaded DiaSource data.
    """
    offset = 0
    connection = legacyApdbUtils.connectToApdb(dbName, dbType, schema)
    try:
        while True:
            sources = pd.read_sql_query(
                f'select * from "DiaSource" ORDER BY ccdVisitId, diaSourceId LIMIT {limit} OFFSET {offset};',
                connection)
            if len(sources) == 0:
                break
            legacyApdbUtils.addTableMetadata(sources,
                                             butler=butler,
                                             instrument=instrument)
            yield sources
            offset += limit
    finally:
        connection.close()


def run_cutouts(args):
    """Run ZooniverseCutoutsTask on the parsed commandline arguments.

    Parameters
    ----------
    args : `argparse.Namespace`
        The parsed commandline arguments.
    """
    # We have to initialize the logger manually on the commandline.
    logging.basicConfig(
        level=logging.INFO, format="{name} {levelname}: {message}", style="{"
    )

    butler = lsst.daf.butler.Butler(args.repo, collections=args.collections)

    config = ZooniverseCutoutsConfig()
    if args.configFile is not None:
        config.load(os.path.expanduser(args.configFile))
    config.freeze()
    cutouts = ZooniverseCutoutsTask(config=config, outputPath=args.outputPath)

    sources = []
    if not args.all:
        data = select_sources(args.dbName, args.dbType, args.schema, butler, args.instrument, args.limit)
        sources = cutouts.run(data, butler, args.outputPath)
    else:
        for data in select_sources(args.dbName,
                                   args.dbType,
                                   args.schema,
                                   butler,
                                   args.instrument,
                                   args.limit):
            sources.extend(cutouts.write_images(data, butler, args.outputPath))
        cutouts.write_manifest(sources, args.outputPath)

    print(f"Generated {len(sources)} diaSource cutouts to {args.outputPath}.")


def main():
    args = build_argparser().parse_args()
    run_cutouts(args)

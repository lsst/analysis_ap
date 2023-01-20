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

"""Construct template/image/difference cutouts for upload to Zooniverse, or
to just to view as images.
"""

__all__ = ["ZooniverseCutoutsConfig", "ZooniverseCutoutsTask", "CutoutPath"]

import argparse
import functools
import io
import itertools
import logging
from math import log10
import multiprocessing
import os
import pathlib

import astropy.units as u
import numpy as np
import pandas as pd

from lsst.ap.association import UnpackApdbFlags
import lsst.dax.apdb
import lsst.pex.config as pexConfig
import lsst.pex.exceptions
import lsst.pipe.base
import lsst.utils

from . import apdb


class ZooniverseCutoutsConfig(pexConfig.Config):
    size = pexConfig.Field(
        doc="Width of cutout to extract for image from science, template, and difference exposures.",
        dtype=int,
        default=30,
    )
    url_root = pexConfig.Field(
        doc="URL that the resulting images will be served to Zooniverse from, for the manifest file. "
            "If not set, no manifest file will be written.",
        dtype=str,
        default=None,
        optional=True,
    )
    diff_image_type = pexConfig.Field(
        doc="Dataset type of template and difference image to use for cutouts; "
            "Will have '_templateExp' and '_differenceExp' appended for butler.get(), respectively.",
        dtype=str,
        default="deepDiff",
    )
    add_metadata = pexConfig.Field(
        doc="Annotate the cutouts with catalog metadata, including coordinates, fluxes, flags, etc.",
        dtype=bool,
        default=False
    )
    chunk_size = pexConfig.Field(
        doc="Chunk up files into subdirectories, with at most this many files per directory."
            " None means write all the files to one `images/` directory.",
        dtype=int,
        default=10000,
        optional=True
    )


class ZooniverseCutoutsTask(lsst.pipe.base.Task):
    """Generate cutouts and a manifest for upload to a Zooniverse project.

    Parameters
    ----------
    output_path : `str`
        The path to write the output to; manifest goes here, while the
        images themselves go into ``output_path/images/``.
    """
    ConfigClass = ZooniverseCutoutsConfig
    _DefaultName = "zooniverseCutouts"

    def __init__(self, *, output_path, **kwargs):
        super().__init__(**kwargs)
        self._output_path = output_path
        self.cutout_path = CutoutPath(output_path, chunk_size=self.config.chunk_size)

    def _reduce_kwargs(self):
        # to allow pickling of this Task
        kwargs = super()._reduce_kwargs()
        kwargs["output_path"] = self._output_path
        return kwargs

    def run(self, data, butler, njobs=0):
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
        njobs : `int`, optional
            Number of multiprocessing jobs to make cutouts with; default of 0
            means don't use multiprocessing at all.

        Returns
        -------
        source_ids : `list` [`int`]
            DiaSourceIds of cutout images that were generated.
        """
        result = self.write_images(data, butler, njobs=njobs)
        self.write_manifest(result)
        self.log.info("Wrote %d images to %s", len(result), self._output_path)
        return result

    def write_manifest(self, sources):
        """Save a Zooniverse manifest attaching image URLs to source ids.

        Parameters
        ----------
        sources : `list` [`int`]
            The diaSourceIds of the sources that had cutouts succesfully made.
        """
        if self.config.url_root is not None:
            manifest = self.make_manifest(sources)
            manifest.to_csv(os.path.join(self._output_path, "manifest.csv"), index=False)
        else:
            self.log.info("No url_root config provided, so no Zooniverse manifest file was written.")

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
        cutout_path = CutoutPath(self.config.url_root)
        manifest = pd.DataFrame()
        manifest["external_id"] = sources
        manifest["location:1"] = [cutout_path(x) for x in sources]
        manifest["metadata:diaSourceId"] = sources
        return manifest

    def write_images(self, data, butler, njobs=0):
        """Make the 3-part cutout images for each requested source and write
        them to disk.

        Creates a ``images/`` subdirectory via cutout_path if one
        does not already exist; images are written there as PNG files.

        Parameters
        ----------
        data : `pandas.DataFrame`
            The DiaSources to extract cutouts for. Must contain at least these
            fields: ``ra, dec, diaSourceId, detector, visit, instrument``.
        butler : `lsst.daf.butler.Butler`
            The butler connection to use to load the data; create it with the
            collections you wish to load images from.
        njobs : `int`, optional
            Number of multiprocessing jobs to make cutouts with; default of 0
            means don't use multiprocessing at all.

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
        pathlib.Path(os.path.join(self._output_path, "images")).mkdir(exist_ok=True)

        sources = []
        if njobs > 0:
            with multiprocessing.Pool(njobs) as pool:
                sources = pool.starmap(self._do_one_source, zip(data.to_records(), flags,
                                                                itertools.repeat(butler)))
        else:
            for i, source in enumerate(data.to_records()):
                id = self._do_one_source(source, flags[i], butler)
                sources.append(id)

        # restore numpy error message state
        np.seterr(**seterr_dict)
        # Only return successful ids, not failures.
        return [s for s in sources if s is not None]

    def _do_one_source(self, source, flags, butler):
        """Make cutouts for one diaSource.

        Parameters
        ----------
        source : `numpy.record`, optional
            DiaSource record for this cutout, to add metadata to the image.
        flags : `str`, optional
            Unpacked bits from the ``flags`` field in ``source``.
            Required if ``source`` is not None.
        butler : `lsst.daf.butler.Butler`
            Butler connection to use to load the data; create it with the
            collections you wish to load images from.

        Returns
        -------
        diaSourceId : `int` or None
            Id of the source that was generated, or None if there was an error.
        """
        @functools.lru_cache(maxsize=4)
        def get_exposures(instrument, detector, visit):
            """Return science, template, difference exposures, using a small
            cache so we don't have to re-read files as often.

            NOTE: Closure because it needs access to the local-scoped butler,
            as lru_cache can't have mutable args in the decorated method.

            If we redo this all to work with BPS or other parallelized
            systems, or get good butler-side caching, we could remove the
            lru_cache above.
            """
            dataId = {'instrument': instrument, 'detector': detector, 'visit': visit}
            return (butler.get('calexp', dataId),
                    butler.get(f'{self.config.diff_image_type}_templateExp', dataId),
                    butler.get(f'{self.config.diff_image_type}_differenceExp', dataId))

        try:
            center = lsst.geom.SpherePoint(source["ra"], source["decl"], lsst.geom.degrees)
            science, template, difference = get_exposures(source["instrument"],
                                                          source["detector"],
                                                          source["visit"])
            scale = science.wcs.getPixelScale().asArcseconds()
            image = self.generate_image(science, template, difference, center, scale,
                                        source=source if self.config.add_metadata else None,
                                        flags=flags if self.config.add_metadata else None)
            self.cutout_path.mkdir(source["diaSourceId"])
            with open(self.cutout_path(source["diaSourceId"]), "wb") as outfile:
                outfile.write(image.getbuffer())
            return source["diaSourceId"]
        except (LookupError, lsst.pex.exceptions.Exception) as e:
            self.log.error(
                f"{e.__class__.__name__} processing diaSourceId {source['diaSourceId']}: {e}"
            )
            return None
        except Exception:
            # Ensure other exceptions are interpretable when multiprocessing.
            import traceback
            traceback.print_exc()
            raise

    def generate_image(self, science, template, difference, center, scale, source=None, flags=None):
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
        scale : `float`
            Pixel scale in arcseconds.
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
            scale,
            source=source,
            flags=flags
        )

    def _plot_cutout(self, science, template, difference, scale, source=None, flags=None):
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
        scale : `float`
            Pixel scale in arcseconds.
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
            ax.imshow(data, cmap=cm.bone, interpolation="none", norm=norm,
                      extent=(0, self.config.size, 0, self.config.size), origin="lower", aspect="equal")
            x_line = 1
            y_line = 1
            ax.plot((x_line, x_line + 1.0/scale), (y_line, y_line), color="blue", lw=6)
            ax.plot((x_line, x_line + 1.0/scale), (y_line, y_line), color="yellow", lw=2)
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

    # rb score
    if np.isfinite(source['spuriousness']):
        fig.text(0.73, 0.79, f"RB:{source['spuriousness']:.03f}",
                 color='#e41a1c' if source['spuriousness'] < 0.5 else '#4daf4a',
                 fontweight="bold")

    fig.text(0.0, 0.79, "total (nJy):", color=flag_color if any(flags[flags_forced]) else text_color)
    fig.text(0.25, 0.79, f"{source['totFlux']:8.1f}", horizontalalignment='right')
    fig.text(0.252, 0.79, "+/-", color=text_color)
    fig.text(0.29, 0.79, f"{source['totFluxErr']:8.1f}")
    fig.text(0.40, 0.79, "S/N:", color=text_color)
    fig.text(0.45, 0.79, f"{abs(source['totFlux']/source['totFluxErr']):6.2f}")
    fig.text(0.55, 0.79, "ABmag:", color=text_color)
    fig.text(0.635, 0.79, f"{(source['totFlux']*u.nanojansky).to_value(u.ABmag):.3f}")


class CutoutPath:
    """Manage paths to image cutouts with filenames based on diaSourceId.

    Supports local files, and id-chunked directories.

    Parameters
    ----------
    root : `str`
        Root file path to manage.
    chunk_size : `int`, optional
        At most this many files per directory. Must be a power of 10.

    Raises
    ------
    RuntimeError
        Raised if chunk_size is not a power of 10.
    """
    def __init__(self, root, chunk_size=None):
        self._root = root
        if chunk_size is not None and (log10(chunk_size) != int(log10(chunk_size))):
            raise RuntimeError(f"CutoutPath file chunk_size must be a power of 10, got {chunk_size}.")
        self._chunk_size = chunk_size

    def __call__(self, id):
        """Return the full path to a diaSource cutout.

        Parameters
        ----------
        id : `int`
            Source id to create the path for.

        Returns
        -------
        path : `str`
            Full path to the requested file.
        """
        def chunker(id, size):
            return (id // size)*size

        if self._chunk_size is not None:
            return os.path.join(self._root, f"images/{chunker(id, self._chunk_size)}/{id}.png")
        else:
            return os.path.join(self._root, f"images/{id}.png")

    def mkdir(self, id):
        """Make the directory tree to write this cutout id to.

        Parameters
        ----------
        id : `int`
            Source id to create the path for.
        """
        path = os.path.dirname(self(id))
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

    apdbArgs = parser.add_mutually_exclusive_group(required=True)
    apdbArgs.add_argument(
        "--sqlitefile",
        default=None,
        help="Path to sqlite file to load from; required for sqlite connection.",
    )
    apdbArgs.add_argument(
        "--namespace",
        default=None,
        help="Postgres namespace (aka schema) to connect to; "
             " required for postgres connections."
    )

    parser.add_argument(
        "--postgres_url",
        default="rubin@usdf-prompt-processing-dev.slac.stanford.edu/lsst-devl",
        help="Postgres connection path, or default (None) to use ApdbPostgresQuery default."
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
        "-j",
        "--jobs",
        default=0,
        type=int,
        help="Number of processes to use when generating cutouts. "
             "Specify 0 (the default) to not use multiprocessing at all."
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


def _make_apdbQuery(butler, instrument, sqlitefile=None, postgres_url=None, namespace=None):
    """Return a query connection to the specified APDB.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler to read detector/visit information from.
    instrument : `lsst.obs.base.Instrument`
        Instrument associated with this data, to get detector/visit data.
    sqlitefile : `str`, optional
        SQLite file to load APDB from; if set, postgres kwargs are ignored.
    postgres_url : `str`, optional
        Postgres connection URL to connect to APDB.
    namespace : `str`, optional
        Postgres schema to load from; required with postgres_url.

    Returns
    -------
    apdb_query : `lsst.analysis.ap.ApdbQuery`
        Query instance to use to load data from APDB.

    Raises
    ------
    RuntimeError
        Raised if the APDB connection kwargs are invalid in some way.
    """
    if sqlitefile is not None:
        apdb_query = apdb.ApdbSqliteQuery(sqlitefile, butler=butler, instrument=instrument)
    elif postgres_url is not None and namespace is not None:
        apdb_query = apdb.ApdbPostgresQuery(namespace, postgres_url, butler=butler, instrument=instrument)
    else:
        raise RuntimeError("Cannot handle database connection args: "
                           f"sqlitefile={sqlitefile}, postgres_url={postgres_url}, namespace={namespace}")
    return apdb_query


def select_sources(apdb_query, limit):
    """Load an APDB and return n sources from it.

    Parameters
    ----------
    apdb_query : `lsst.analysis.ap.ApdbQuery`
        APDB query interface to load from.
    limit : `int`
        Number of sources to select from the APDB.

    Returns
    -------
    sources : `pandas.DataFrame`
        The loaded DiaSource data.
    """
    offset = 0
    try:
        while True:
            with apdb_query.connection as connection:
                sources = pd.read_sql_query(
                    'select * FROM "DiaSource" ORDER BY "ccdVisitId", '
                    f'"diaSourceId" LIMIT {limit} OFFSET {offset};',
                    connection)
            if len(sources) == 0:
                break
            apdb_query._fill_from_ccdVisitId(sources)

            yield sources
            offset += limit
    finally:
        connection.close()


def len_sources(apdb_query):
    """Return the number of DiaSources in the supplied APDB.

    Parameters
    ----------
    apdb_query : `lsst.analysis.ap.ApdbQuery`
        APDB query interface to load from.

    Returns
    -------
    count : `int`
        Number of diaSources in this APDB.
    """
    with apdb_query.connection as connection:
        cursor = connection.cursor()
        cursor.execute('select count(*) FROM "DiaSource";')
        count = cursor.fetchone()[0]
    return count


def run_cutouts(args):
    """Run ZooniverseCutoutsTask on the parsed commandline arguments.

    Parameters
    ----------
    args : `argparse.Namespace`
        Parsed commandline arguments.
    """
    # We have to initialize the logger manually on the commandline.
    logging.basicConfig(
        level=logging.INFO, format="{name} {levelname}: {message}", style="{"
    )

    butler = lsst.daf.butler.Butler(args.repo, collections=args.collections)
    apdb_query = _make_apdbQuery(butler,
                                 args.instrument,
                                 sqlitefile=args.sqlitefile,
                                 postgres_url=args.postgres_url,
                                 namespace=args.namespace)
    data = select_sources(apdb_query, args.limit)

    config = ZooniverseCutoutsConfig()
    if args.configFile is not None:
        config.load(os.path.expanduser(args.configFile))
    config.freeze()
    cutouts = ZooniverseCutoutsTask(config=config, output_path=args.outputPath)

    getter = select_sources(apdb_query, args.limit)
    # Process just one block of length "limit", or all sources in the database?
    if not args.all:
        data = next(getter)
        sources = cutouts.run(data, butler, njobs=args.jobs)
    else:
        sources = []
        count = len_sources(apdb_query)
        for i, data in enumerate(getter):
            sources.extend(cutouts.write_images(data, butler, njobs=args.jobs))
            print(f"Completed {i} batches of {args.limit} size, out of {count} diaSources.")
        cutouts.write_manifest(sources)

    print(f"Generated {len(sources)} diaSource cutouts to {args.outputPath}.")


def main():
    args = build_argparser().parse_args()
    run_cutouts(args)

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

__all__ = ["PlotImageSubtractionCutoutsConfig", "PlotImageSubtractionCutoutsTask", "CutoutPath"]

import argparse
import functools
import io
import logging
import multiprocessing
import os
from math import log10

import astropy.units as u
from lsst.daf.butler import DatasetNotFoundError
import lsst.dax.apdb
import lsst.geom
import lsst.pex.config as pexConfig
import lsst.pex.exceptions
import lsst.pipe.base
import lsst.utils
import numpy as np
import pandas as pd
import sqlalchemy

from . import apdb


class _ButlerCache:
    """Global class to handle butler queries, to allow lru_cache and
    `multiprocessing.Pool` to work together.

    If we redo this all to work with BPS or other parallelized systems, or get
    good butler-side caching, we could remove this lru_cache system.
    """

    def set(self, butler, config):
        """Call this to store a Butler and Config instance before using the
        global class instance.

        Parameters
        ----------
        butler : `lsst.daf.butler.Butler`
            Butler instance to store.
        config : `lsst.pex.config.Config`
            Config instance to store.
        """
        self._butler = butler
        self._config = config
        # Ensure the caches are empty if we've been re-set.
        self.get_exposures.cache_clear()
        self.get_catalog.cache_clear()

    @functools.lru_cache(maxsize=4)
    def get_exposures(self, instrument, detector, visit):
        """Return science, template, difference exposures, using a small
        cache so we don't have to re-read files as often.

        Parameters
        ----------
        instrument : `str`
            Instrument name to define the data id.
        detector : `int`
            Detector id to define the data id.
        visit : `int`
            Visit id to define the data id.

        Returns
        -------
        exposures : `tuple` [`lsst.afw.image.ExposureF`]
            Science, template, and difference exposure for this data id.
        """
        data_id = {'instrument': instrument, 'detector': detector, 'visit': visit}
        try:
            self._butler.get(self._config.science_image_type, data_id)
        except DatasetNotFoundError as e:
            self.log.error(f'Cannot load {self._config.science_image_type} with data_id {data_id}: {e}')
            if self._config.science_image_type == 'calexp':
                self.log.info(f'No {self._config.science_image_type} found, trying initial_pvi')
                self._config.science_image_type = 'initial_pvi'
            elif self._config.science_image_type == 'initial_pvi':
                self.log.info(f'No {self._config.science_image_type} found, trying calexp')
                self._config.science_image_type = 'calexp'
            else:
                self.log.info('Must provide a valid datasetType and dataId to retrieve science image.')
        finally:
            return (self._butler.get(self._config.science_image_type, data_id),
                    self._butler.get(f'{self._config.diff_image_type}_templateExp', data_id),
                    self._butler.get(f'{self._config.diff_image_type}_differenceExp', data_id))

    @functools.lru_cache(maxsize=4)
    def get_catalog(self, instrument, detector, visit):
        """Return the diaSrc catalog from the butler.

        Parameters
        ----------
        instrument : `str`
            Instrument name to define the data id.
        detector : `int`
            Detector id to define the data id.
        visit : `int`
            Visit id to define the data id.

        Returns
        -------
        catalog : `lsst.afw.table.SourceCatalog`
            DiaSource catalog for this data id.
            """
        data_id = {'instrument': instrument, 'detector': detector, 'visit': visit}
        return self._butler.get(f'{self._config.diff_image_type}_diaSrc', data_id)


# Global used within each multiprocessing worker (or single process).
butler_cache = _ButlerCache()


class PlotImageSubtractionCutoutsConfig(pexConfig.Config):
    sizes = pexConfig.ListField(
        doc="List of widths of cutout to extract for image from science, \
            template, and difference exposures.",
        dtype=int,
        default=[30],
    )
    use_footprint = pexConfig.Field(
        doc="Use source footprint to to define cutout region; "
            "If set, ignore `size` and use the footprint bbox instead.",
        dtype=bool,
        default=False,
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
        default="goodSeeingDiff",
    )
    science_image_type = pexConfig.Field(
        doc="Dataset type of science image to use for cutouts.",
        dtype=str,
        default="calexp",
    )
    add_metadata = pexConfig.Field(
        doc="Annotate the cutouts with catalog metadata, including coordinates, fluxes, flags, etc.",
        dtype=bool,
        default=True
    )
    chunk_size = pexConfig.Field(
        doc="Chunk up files into subdirectories, with at most this many files per directory."
            " None means write all the files to one `images/` directory.",
        dtype=int,
        default=10000,
        optional=True
    )
    save_as_numpy = pexConfig.Field(
        doc="Save the raw cutout images in numpy format.",
        dtype=bool,
        default=False
    )


class PlotImageSubtractionCutoutsTask(lsst.pipe.base.Task):
    """Generate template/science/difference image cutouts of DiaSources and an
    optional manifest for upload to a Zooniverse project.

    Parameters
    ----------
    output_path : `str`
        The path to write the output to; manifest goes here, while the
        images themselves go into ``output_path/images/``.
    """
    ConfigClass = PlotImageSubtractionCutoutsConfig
    _DefaultName = "plotImageSubtractionCutouts"

    def __init__(self, *, output_path, **kwargs):
        super().__init__(**kwargs)
        self._output_path = output_path
        self.cutout_path = CutoutPath(output_path, chunk_size=self.config.chunk_size)
        self.numpy_path = CutoutPath(output_path, chunk_size=self.config.chunk_size,
                                     subdirectory='numpy')

    def _reduce_kwargs(self):
        # to allow pickling of this Task
        kwargs = super()._reduce_kwargs()
        kwargs["output_path"] = self._output_path
        return kwargs

    def run(self, data, butler, njobs=0):
        """Generate cutout images and a manifest for upload to Zooniverse
        from a collection of DiaSources.

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
            manifest = self._make_manifest(sources)
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
        manifest["location:1"] = [cutout_path(x, f'{x}.png') for x in sources]
        manifest["metadata:diaSourceId"] = sources
        return manifest

    def write_images(self, data, butler, njobs=0):
        """Make the 3-part cutout images for each requested source and write
        them to disk.

        Creates ``images/`` and ``numpy/`` subdirectories if they
        do not already exist; images are written there as PNG and npy files.

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

        # Exclude index if they are replicated in columns.
        indexNotInColumns = not any(index in data.columns for index in data.index.names)

        sources = []
        butler_cache.set(butler, self.config)
        if njobs > 0:
            with multiprocessing.Pool(njobs) as pool:
                sources = pool.map(self._do_one_source, data.to_records(index=indexNotInColumns))
        else:
            for i, source in enumerate(data.to_records(index=indexNotInColumns)):
                if not self.cutout_path.exists(source["diaSourceId"],
                                               f'{source["diaSourceId"]}.png'):
                    id = self._do_one_source(source)
                sources.append(id)

        # restore numpy error message state
        np.seterr(**seterr_dict)
        # Only return successful ids, not failures.
        return [s for s in sources if s is not None]

    def _do_one_source(self, source):
        """Make cutouts for one diaSource.

        Parameters
        ----------
        source : `numpy.record`, optional
            DiaSource record for this cutout, to add metadata to the image.

        Returns
        -------
        diaSourceId : `int` or None
            Id of the source that was generated, or None if there was an error.
        """
        try:
            center = lsst.geom.SpherePoint(source["ra"], source["dec"], lsst.geom.degrees)
            science, template, difference = butler_cache.get_exposures(source["instrument"],
                                                                       source["detector"],
                                                                       source["visit"])
            if self.config.use_footprint:
                catalog = butler_cache.get_catalog(source["instrument"],
                                                   source["detector"],
                                                   source["visit"])
                # The input catalogs must be sorted.
                if not catalog.isSorted():
                    data_id = {'instrument': source["instrument"],
                               'detector': source["detector"],
                               'visit': source["visit"]}
                    msg = f"{self.config.diff_image_type}_diaSrc catalog for {data_id} is not sorted!"
                    raise RuntimeError(msg)
                record = catalog.find(source['diaSourceId'])
                footprint = record.getFootprint()

            scale = science.wcs.getPixelScale(science.getBBox().getCenter()).asArcseconds()
            image = self.generate_image(science, template, difference, center, scale,
                                        dia_source_id=source['diaSourceId'],
                                        save_as_numpy=self.config.save_as_numpy,
                                        source=source if self.config.add_metadata else None,
                                        footprint=footprint if self.config.use_footprint else None)
            self.cutout_path.mkdir(source["diaSourceId"])
            with open(self.cutout_path(source["diaSourceId"],
                      f'{source["diaSourceId"]}.png'), "wb") as outfile:
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

    def generate_image(self, science, template, difference, center, scale, dia_source_id=None,
                       save_as_numpy=False, source=None, footprint=None):
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
        dia_source_id : `int`, optional
            DiaSourceId to use in the filename, if saving to disk.
        save_as_numpy : `bool`, optional
            Save the raw cutout images in numpy format.
        source : `numpy.record`, optional
            DiaSource record for this cutout, to add metadata to the image.
        footprint : `lsst.afw.detection.Footprint`, optional
            Detected source footprint; if specified, extract a square
            surrounding the footprint bbox, otherwise use ``config.size``.

        Returns
        -------
        image: `io.BytesIO`
            The generated image, to be output to a file or displayed on screen.
        """
        numpy_cutouts = {}
        if not self.config.use_footprint:
            sizes = self.config.sizes
            cutout_science, cutout_template, cutout_difference = [], [], []
            for i, s in enumerate(sizes):
                extent = lsst.geom.Extent2I(s, s)
                science_cutout = science.getCutout(center, extent)
                template_cutout = template.getCutout(center, extent)
                difference_cutout = difference.getCutout(center, extent)
                if save_as_numpy:
                    self.numpy_path.mkdir(dia_source_id)
                    numpy_cutouts[f"sci_{s}"] = science_cutout.image.array
                    numpy_cutouts[f"temp_{s}"] = template_cutout.image.array
                    numpy_cutouts[f"diff_{s}"] = difference_cutout.image.array
                    for cutout_type, cutout in numpy_cutouts.items():
                        outfile = self.numpy_path(dia_source_id, f'{dia_source_id}_{cutout_type}.npy')
                        np.save(outfile, np.expand_dims(cutout, axis=0))
                cutout_science.append(science_cutout)
                cutout_template.append(template_cutout)
                cutout_difference.append(difference_cutout)
        else:
            if self.config.save_as_numpy:
                raise RuntimeError("Cannot save as numpy when using footprints.")
            cutout_science = [science.getCutout(footprint.getBBox())]
            cutout_template = [template.getCutout(footprint.getBBox())]
            cutout_difference = [difference.getCutout(footprint.getBBox())]
            extent = footprint.getBBox().getDimensions()
            # Plot a square equal to the largest dimension.
            sizes = [extent.x if extent.x > extent.y else extent.y]

        return self._plot_cutout(cutout_science,
                                 cutout_template,
                                 cutout_difference,
                                 scale,
                                 sizes,
                                 source=source)

    def _plot_cutout(self, science, template, difference, scale, sizes, source=None):
        """Plot the cutouts for a source in one image.

        Parameters
        ----------
        science : `list` [`lsst.afw.image.ExposureF`]
            List of cutout Science exposure(s) to include in the image.
        template : `list` [`lsst.afw.image.ExposureF`]
            List of cutout template exposure(s) to include in the image.
        difference : `list` [`lsst.afw.image.ExposureF`]
            List of cutout science minus template exposure(s) to include
            in the image.
        source : `numpy.record`, optional
            DiaSource record for this cutout, to add metadata to the image.
        scale : `float`
            Pixel scale in arcseconds.
        size : `list` [`int`]
            List of x/y dimensions of of the images passed in, to set imshow
            extent.

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

        def plot_one_image(ax, data, size, name=None):
            """Plot a normalized image on an axis."""
            if name == "Difference":
                norm = aviz.ImageNormalize(
                    # focus on a rect of dim 15 at the center of the image.
                    data[data.shape[0] // 2 - 7:data.shape[0] // 2 + 8,
                         data.shape[1] // 2 - 7:data.shape[1] // 2 + 8],
                    interval=aviz.MinMaxInterval(),
                    stretch=aviz.AsinhStretch(a=0.1),
                )
            else:
                norm = aviz.ImageNormalize(
                    data,
                    interval=aviz.MinMaxInterval(),
                    stretch=aviz.AsinhStretch(a=0.1),
                )
            ax.imshow(data, cmap=cm.bone, interpolation="none", norm=norm,
                      extent=(0, size, 0, size), origin="lower", aspect="equal")
            x_line = 1
            y_line = 1
            ax.plot((x_line, x_line + 1.0/scale), (y_line, y_line), color="blue", lw=6)
            ax.plot((x_line, x_line + 1.0/scale), (y_line, y_line), color="yellow", lw=2)
            ax.axis("off")
            if name is not None:
                ax.set_title(name)

        try:
            len_sizes = len(sizes)
            fig, axs = plt.subplots(len_sizes, 3, constrained_layout=True)
            if len_sizes == 1:
                plot_one_image(axs[0], template[0].image.array, sizes[0], "Template")
                plot_one_image(axs[1], science[0].image.array, sizes[0], "Science")
                plot_one_image(axs[2], difference[0].image.array, sizes[0], "Difference")
            else:
                plot_one_image(axs[0][0], template[0].image.array, sizes[0], "Template")
                plot_one_image(axs[0][1], science[0].image.array, sizes[0], "Science")
                plot_one_image(axs[0][2], difference[0].image.array, sizes[0], "Difference")
                for i in range(1, len(axs)):
                    plot_one_image(axs[i][0], template[i].image.array, sizes[i], None)
                    plot_one_image(axs[i][1], science[i].image.array, sizes[i], None)
                    plot_one_image(axs[i][2], difference[i].image.array, sizes[i], None)
            if source is not None:
                _annotate_image(fig, source, len_sizes)

            output = io.BytesIO()
            plt.savefig(output, bbox_inches="tight", format="png")
            output.seek(0)  # to ensure opening the image starts from the front
        finally:
            plt.close(fig)

        return output


def _annotate_image(fig, source, len_sizes):
    """Annotate the cutouts image with metadata and flags.

    Parameters
    ----------
    fig : `matplotlib.Figure`
        Figure to be annotated.
    source : `numpy.record`
        DiaSource record of the object being plotted.
    len_sizes : `int`
        Length of the ``size`` array set in configuration.
    """
    # Names of flags fields to add a flag label to the image, using any().
    flags_psf = ["psfFlux_flag", "psfFlux_flag_noGoodPixels", "psfFlux_flag_edge"]
    flags_aperture = ["apFlux_flag", "apFlux_flag_apertureTruncated"]
    flags_forced = ["forced_PsfFlux_flag", "forced_PsfFlux_flag_noGoodPixels",
                    "forced_PsfFlux_flag_edge"]
    flags_edge = ["pixelFlags_edge"]
    flags_interp = ["pixelFlags_interpolated", "pixelFlags_interpolatedCenter"]
    flags_saturated = ["pixelFlags_saturated", "pixelFlags_saturatedCenter"]
    flags_cr = ["pixelFlags_cr", "pixelFlags_crCenter"]
    flags_bad = ["pixelFlags_bad"]
    flags_suspect = ["pixelFlags_suspect", "pixelFlags_suspectCenter"]
    flags_centroid = ["centroid_flag"]
    flags_shape = ["shape_flag", "shape_flag_no_pixels", "shape_flag_not_contained",
                   "shape_flag_parent_source"]

    flag_color = "red"
    text_color = "grey"

    if len_sizes == 1:
        heights = [0.95, 0.91, 0.87, 0.83, 0.79]
    else:
        heights = [1.2, 1.15, 1.1, 1.05, 1.0]

    # NOTE: fig.text coordinates are in fractions of the figure.
    fig.text(0, heights[0], "diaSourceId:", color=text_color)
    fig.text(0.145, heights[0], f"{source['diaSourceId']}")
    fig.text(0.43, heights[0], f"{source['instrument']}", fontweight="bold")
    fig.text(0.64, heights[0], "detector:", color=text_color)
    fig.text(0.74, heights[0], f"{source['detector']}")
    fig.text(0.795, heights[0], "visit:", color=text_color)
    fig.text(0.85, heights[0], f"{source['visit']}")
    fig.text(0.95, heights[0], f"{source['band']}")

    fig.text(0.0, heights[1], "ra:", color=text_color)
    fig.text(0.037, heights[1], f"{source['ra']:.8f}")
    fig.text(0.21, heights[1], "dec:", color=text_color)
    fig.text(0.265, heights[1], f"{source['dec']:+.8f}")
    fig.text(0.50, heights[1], "detection S/N:", color=text_color)
    fig.text(0.66, heights[1], f"{source['snr']:6.1f}")
    fig.text(0.75, heights[1], "PSF chi2:", color=text_color)
    fig.text(0.85, heights[1], f"{source['psfChi2']/source['psfNdata']:6.2f}")

    fig.text(0.0, heights[2], "PSF (nJy):", color=flag_color if any(source[flags_psf]) else text_color)
    fig.text(0.25, heights[2], f"{source['psfFlux']:8.1f}", horizontalalignment='right')
    fig.text(0.252, heights[2], "+/-", color=text_color)
    fig.text(0.29, heights[2], f"{source['psfFluxErr']:8.1f}")
    fig.text(0.40, heights[2], "S/N:", color=text_color)
    fig.text(0.45, heights[2], f"{abs(source['psfFlux']/source['psfFluxErr']):6.2f}")

    # NOTE: yellow is hard to read on white; use goldenrod instead.
    if any(source[flags_edge]):
        fig.text(0.55, heights[2], "EDGE", color="goldenrod", fontweight="bold")
    if any(source[flags_interp]):
        fig.text(0.62, heights[2], "INTERP", color="green", fontweight="bold")
    if any(source[flags_saturated]):
        fig.text(0.72, heights[2], "SAT", color="green", fontweight="bold")
    if any(source[flags_cr]):
        fig.text(0.77, heights[2], "CR", color="magenta", fontweight="bold")
    if any(source[flags_bad]):
        fig.text(0.81, heights[2], "BAD", color="red", fontweight="bold")
    if source['isDipole']:
        fig.text(0.87, heights[2], "DIPOLE", color="indigo", fontweight="bold")

    fig.text(0.0, heights[3], "ap (nJy):", color=flag_color if any(source[flags_aperture]) else text_color)
    fig.text(0.25, heights[3], f"{source['apFlux']:8.1f}", horizontalalignment='right')
    fig.text(0.252, heights[3], "+/-", color=text_color)
    fig.text(0.29, heights[3], f"{source['apFluxErr']:8.1f}")
    fig.text(0.40, heights[3], "S/N:", color=text_color)
    fig.text(0.45, heights[3], f"{abs(source['apFlux']/source['apFluxErr']):#6.2f}")

    if any(source[flags_suspect]):
        fig.text(0.55, heights[3], "SUS", color="goldenrod", fontweight="bold")
    if any(source[flags_centroid]):
        fig.text(0.60, heights[3], "CENTROID", color="red", fontweight="bold")
    if any(source[flags_shape]):
        fig.text(0.73, heights[3], "SHAPE", color="red", fontweight="bold")
    # Future option: to add two more flag flavors to the legend,
    # use locations 0.80 and 0.87

    # rb score
    if source['reliability'] is not None and np.isfinite(source['reliability']):
        fig.text(0.73, heights[4], f"RB:{source['reliability']:.03f}",
                 color='#e41a1c' if source['reliability'] < 0.5 else '#4daf4a',
                 fontweight="bold")

    fig.text(0.0, heights[4], "sci (nJy):", color=flag_color if any(source[flags_forced]) else text_color)
    fig.text(0.25, heights[4], f"{source['scienceFlux']:8.1f}", horizontalalignment='right')
    fig.text(0.252, heights[4], "+/-", color=text_color)
    fig.text(0.29, heights[4], f"{source['scienceFluxErr']:8.1f}")
    fig.text(0.40, heights[4], "S/N:", color=text_color)
    fig.text(0.45, heights[4], f"{abs(source['scienceFlux']/source['scienceFluxErr']):6.2f}")
    fig.text(0.55, heights[4], "ABmag:", color=text_color)
    fig.text(0.635, heights[4], f"{(source['scienceFlux']*u.nanojansky).to_value(u.ABmag):.3f}")


class CutoutPath:
    """Manage paths to image cutouts with filenames based on diaSourceId.

    Supports local files, and id-chunked directories.

    Parameters
    ----------
    root : `str`
        Root file path to manage.
    chunk_size : `int`, optional
        At most this many files per directory. Must be a power of 10.
    subdirectory : `str`, optional
        Name of the subdirectory

    Raises
    ------
    RuntimeError
        Raised if chunk_size is not a power of 10.
    """

    def __init__(self, root, chunk_size=None, subdirectory='images'):
        self._root = root
        if chunk_size is not None and (log10(chunk_size) != int(log10(chunk_size))):
            raise RuntimeError(f"CutoutPath file chunk_size must be a power of 10, got {chunk_size}.")
        self._chunk_size = chunk_size
        self._subdirectory = subdirectory

    def directory(self, id):
        """Return the directory to store the output in.

        Parameters
        ----------
        id : `int`
            Source id to create the path for.

        Returns
        -------
        directory: `str`
            Directory for this file.
        """

        def chunker(id, size):
            return (id // size)*size

        if self._chunk_size is not None:
            return os.path.join(self._root,
                                f"{self._subdirectory}/{chunker(id, self._chunk_size)}")
        else:
            return os.path.join(self._root, f"{self._subdirectory}")

    def __call__(self, id, filename):
        """Return the full path to a diaSource cutout.

        Parameters
        ----------
        id : `int`
            Source id to create the path for.
        filename: `str`
            Filename to write.

        Returns
        -------
        path : `str`
            Full path to the requested file.
        """

        return os.path.join(self.directory(id), filename)

    def exists(self, id, filename):
        """Return True if the file already exists.

        Parameters
        ----------
        id : `int`
            Source id to create the path for.
        filename: `str`
            Filename to write.

        Returns
        -------
        exists : `bool`
            Does the supplied filename exist?
        """

        return os.path.exists(os.path.join(self.directory(id), filename))

    def mkdir(self, id):
        """Make the directory tree to write this cutout id to.

        Parameters
        ----------
        id : `int`
            Source id to create the path for.
        """
        os.makedirs(self.directory(id), exist_ok=True)


def build_argparser():
    """Construct an argument parser for the ``plotImageSubtractionCutouts``
    script.

    Returns
    -------
    argparser : `argparse.ArgumentParser`
        The argument parser that defines the ``plotImageSubtractionCutouts``
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
             "number of sources to load per 'page' when `--all` is set. "
             "This should be significantly larger (100x or more) than the value of `-j`, "
             "to ensure efficient use of each process.",
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
             "Specify 0 (the default) to not use multiprocessing at all. "
             "Note that `--limit` determines how efficiently each process is filled."
    )

    parser.add_argument(
        "-C",
        "--configFile",
        help="File containing the PlotImageSubtractionCutoutsConfig to load.",
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
    parser.add_argument(
        "--reliabilityMin",
        type=float,
        default=None,
        help="Minimum reliability value (default=None) on which to filter the DiaSources.",
    )
    parser.add_argument(
        "--reliabilityMax",
        type=float,
        default=None,
        help="Maximum reliability value (default=None) on which to filter the DiaSources.",
    )
    return parser


def _make_apdbQuery(sqlitefile=None, postgres_url=None, namespace=None):
    """Return a query connection to the specified APDB.

    Parameters
    ----------
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
        apdb_query = apdb.ApdbSqliteQuery(sqlitefile)
    elif postgres_url is not None and namespace is not None:
        apdb_query = apdb.ApdbPostgresQuery(namespace, postgres_url)
    else:
        raise RuntimeError("Cannot handle database connection args: "
                           f"sqlitefile={sqlitefile}, postgres_url={postgres_url}, namespace={namespace}")
    return apdb_query


def select_sources(apdb_query, limit, reliabilityMin=None, reliabilityMax=None):
    """Load an APDB and return n sources from it.

    Parameters
    ----------
    apdb_query : `lsst.analysis.ap.ApdbQuery`
        APDB query interface to load from.
    limit : `int`
        Number of sources to select from the APDB.
    reliabilityMin : `float`
        Minimum reliability value on which to filter the DiaSources.
    reliabilityMax : `float`
        Maximum reliability value on which to filter the DiaSources.

    Returns
    -------
    sources : `pandas.DataFrame`
        The loaded DiaSource data.
    """
    offset = 0
    try:
        while True:
            with apdb_query.connection as connection:
                table = apdb_query._tables["DiaSource"]
                query = table.select()
                if reliabilityMin is not None:
                    query = query.where(table.columns['reliability'] >= reliabilityMin)
                if reliabilityMax is not None:
                    query = query.where(table.columns['reliability'] <= reliabilityMax)
                query = query.order_by(table.columns["visit"],
                                       table.columns["detector"],
                                       table.columns["diaSourceId"])
                query = query.limit(limit).offset(offset)
                sources = pd.read_sql_query(query, connection)
            if len(sources) == 0:
                break
            apdb_query._fill_from_instrument(sources)

            yield sources
            offset += limit
    finally:
        connection.close()


def len_sources(apdb_query, namespace=None):
    """Return the number of DiaSources in the supplied APDB.

    Parameters
    ----------
    apdb_query : `lsst.analysis.ap.ApdbQuery`
        APDB query interface to load from.
    namespace : `str`, optional
        Postgres schema to load data from.

    Returns
    -------
    count : `int`
        Number of diaSources in this APDB.
    """
    with apdb_query.connection as connection:
        if namespace:
            connection.execute(sqlalchemy.text(f"SET search_path TO {namespace}"))
        count = connection.execute(sqlalchemy.text('select count(*) FROM "DiaSource";')).scalar()
    return count


def run_cutouts(args):
    """Run PlotImageSubtractionCutoutsTask on the parsed commandline arguments.

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
    apdb_query = _make_apdbQuery(sqlitefile=args.sqlitefile,
                                 postgres_url=args.postgres_url,
                                 namespace=args.namespace)

    config = PlotImageSubtractionCutoutsConfig()
    if args.configFile is not None:
        config.load(os.path.expanduser(args.configFile))
    config.freeze()
    cutouts = PlotImageSubtractionCutoutsTask(config=config, output_path=args.outputPath)

    if config.save_as_numpy:
        # save the RB output up front so we can use partial runs
        data = select_sources(apdb_query, args.limit, args.reliabilityMin, args.reliabilityMax)
        cols_to_export = ["diaSourceId", "visit", "detector", "diaObjectId",
                          "ssObjectId", "midpointMjdTai", "ra", "dec", "x", "y",
                          "apFlux", "apFluxErr", "snr", "psfFlux", "psfFluxErr",
                          "isDipole", "trailLength", "band", "extendedness",
                          "pixelFlags_bad", "pixelFlags_cr", "pixelFlags_crCenter",
                          "pixelFlags_edge", "pixelFlags_interpolated", "pixelFlags_interpolatedCenter",
                          "pixelFlags_offimage", "pixelFlags_saturated", "pixelFlags_saturatedCenter",
                          "pixelFlags_suspect", "pixelFlags_suspectCenter", "pixelFlags_streak",
                          "pixelFlags_streakCenter", "pixelFlags_injected", "pixelFlags_injectedCenter",
                          "pixelFlags_injected_template", "pixelFlags_injected_templateCenter"]
        # this is inefficient but otherwise we don't use the same query
        all_data = pd.concat([d[cols_to_export] for d in data])
        all_data.to_csv(os.path.join(args.outputPath, "all_diasources.csv.gz"), index=False)

    getter = select_sources(apdb_query, args.limit, args.reliabilityMin, args.reliabilityMax)
    # Process just one block of length "limit", or all sources in the database?
    if not args.all:
        data = next(getter)
        sources = cutouts.run(data, butler, njobs=args.jobs)
    else:
        sources = []
        count = len_sources(apdb_query, args.namespace)
        for i, data in enumerate(getter):
            sources.extend(cutouts.write_images(data, butler, njobs=args.jobs))
            print(f"Completed {i+1} batches of {args.limit} size, out of {count} diaSources.")
        cutouts.write_manifest(sources)

    if config.save_as_numpy:
        # Write a dataframe with only diasources successfully written.
        data.loc[data['diaSourceId'].isin(sources), cols_to_export].to_csv(
            os.path.join(args.outputPath, "exported_diasources.csv.gz"), index=False)

    print(f"Generated {len(sources)} diaSource cutouts to {args.outputPath}.")


def main():
    args = build_argparser().parse_args()
    run_cutouts(args)

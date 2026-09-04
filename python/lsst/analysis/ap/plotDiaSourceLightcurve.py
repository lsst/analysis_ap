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

"""Render diaSource cutouts with a per-diaObject lightcurve panel.

``PlotDiaSourceLightcurveTask`` extends ``PlotImageSubtractionCutoutsTask``
by adding a lightcurve panel below the science/template/difference cutouts.
The panel shows the diaSource history for the associated diaObject and
overlays diaForcedSource measurements for any visit that has a forced
measurement but no diaSource (drawn with a distinct marker).
"""

__all__ = ["PlotDiaSourceLightcurveConfig", "PlotDiaSourceLightcurveTask"]

import argparse
import io
import logging
import os

import lsst.daf.butler
import lsst.pex.config as pexConfig

import pandas as pd

from . import apdb as _apdb_mod
from . import plotUtils
from .plotImageSubtractionCutouts import (
    PlotImageSubtractionCutoutsConfig,
    PlotImageSubtractionCutoutsTask,
    _annotate_image,
)

_log = logging.getLogger(__name__)


class PlotDiaSourceLightcurveConfig(PlotImageSubtractionCutoutsConfig):
    lightcurve_height = pexConfig.Field(
        doc="Height in inches reserved for the lightcurve panel below the cutouts.",
        dtype=float,
        default=2.5,
    )
    lightcurve_exclude_flagged = pexConfig.Field(
        doc="Pass exclude_flagged=True to the APDB query when loading "
            "diaSources for the lightcurve. Defaults to False so the "
            "lightcurve matches the row count of a direct APDB query; "
            "set True to drop diaSources matching the configured bad-flag "
            "list. DiaForcedSources are always loaded unfiltered.",
        dtype=bool,
        default=False,
    )
    lightcurve_marker_source = pexConfig.Field(
        doc="Matplotlib marker style for visits that have a diaSource "
            "detection.",
        dtype=str,
        default="o",
    )
    lightcurve_marker_forced_only = pexConfig.Field(
        doc="Matplotlib marker style for visits that have a forced "
            "measurement but no diaSource.",
        dtype=str,
        default="v",
    )
    highlight_current_source = pexConfig.Field(
        doc="Draw a vertical line and open ring on the lightcurve at the "
            "MJD/flux of the diaSource being cut out.",
        dtype=bool,
        default=True,
    )


class PlotDiaSourceLightcurveTask(PlotImageSubtractionCutoutsTask):
    """Generate cutouts plus a diaObject lightcurve panel for each diaSource.

    Parameters
    ----------
    output_path : `str`
        Path to write outputs to. Same convention as the parent task.
    apdb_query : `lsst.analysis.ap.apdb.DbQuery`, optional
        Query handle used to load the diaSource and diaForcedSource history
        for each diaObject. If None, the lightcurve panel is rendered with
        a placeholder message.

    Notes
    -----
    The input ``data`` DataFrame must include the fields required by the
    parent (``ra, dec, diaSourceId, detector, visit, instrument``) plus
    ``diaObjectId`` (to load the lightcurve) and ``midpointMjdTai`` (to
    highlight the current source on the lightcurve).
    """
    ConfigClass = PlotDiaSourceLightcurveConfig
    _DefaultName = "plotDiaSourceLightcurve"

    def __init__(self, *, output_path, apdb_query=None, **kwargs):
        super().__init__(output_path=output_path, **kwargs)
        self._apdb_query = apdb_query
        # Per-diaObject lightcurve cache so adjacent diaSources on the same
        # object don't re-query the APDB. Keyed by diaObjectId.
        self._lightcurve_cache = {}

    def _reduce_kwargs(self):
        kwargs = super()._reduce_kwargs()
        kwargs["apdb_query"] = self._apdb_query
        return kwargs

    def run(self, data, butler, njobs=0):
        if njobs > 0:
            self.log.warning("njobs=%d ignored; PlotDiaSourceLightcurveTask "
                             "runs single-process only.", njobs)
        return super().run(data, butler, njobs=0)

    def write_images(self, data, butler, njobs=0):
        if njobs > 0:
            self.log.warning("njobs=%d ignored; PlotDiaSourceLightcurveTask "
                             "runs single-process only.", njobs)
        return super().write_images(data, butler, njobs=0)

    def _load_lightcurve_data(self, dia_object_id):
        """Return cached (sources, forced) for one diaObject.

        Returns ``(None, None)`` if no APDB query handle is configured.
        """
        if self._apdb_query is None:
            return None, None
        if dia_object_id not in self._lightcurve_cache:
            sources = self._apdb_query.load_sources_for_object(
                dia_object_id,
                exclude_flagged=self.config.lightcurve_exclude_flagged,
            )
            # Always show all of the forced sources regardless of flags.
            forced = self._apdb_query.load_forced_sources_for_object(
                dia_object_id,
            )
            self._lightcurve_cache[dia_object_id] = (sources, forced)
        return self._lightcurve_cache[dia_object_id]

    def _plot_cutout(self, science, template, difference, scale, sizes, source=None):
        import astropy.visualization as aviz
        import matplotlib
        matplotlib.use("AGG")
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.gridspec import GridSpec

        len_sizes = len(sizes)

        sources_lc, forced_lc = (None, None)
        if source is not None and "diaObjectId" in source.dtype.names:
            dia_object_id = source["diaObjectId"]
            # diaObjectId is NaN for diaSources not associated with any
            # diaObject (e.g. single unassociated detections). Skip the
            # lightcurve query — the panel will render its empty placeholder.
            if not pd.isna(dia_object_id):
                try:
                    sources_lc, forced_lc = self._load_lightcurve_data(int(dia_object_id))
                except Exception as e:
                    self.log.warning("Failed to load lightcurve for diaObjectId=%s: %s. "
                                     "The DiaSource is likely unassociated.",
                                     dia_object_id, e)

        cutout_height_in = max(1.7, 1.7 * len_sizes)
        lc_height_in = float(self.config.lightcurve_height)
        fig_height = cutout_height_in + lc_height_in
        fig = plt.figure(figsize=(7, fig_height), constrained_layout=True)

        gs = GridSpec(2, 1, height_ratios=[cutout_height_in, lc_height_in], figure=fig)
        cutout_gs = gs[0].subgridspec(len_sizes, 3)
        cutout_axes = [[fig.add_subplot(cutout_gs[r, c]) for c in range(3)]
                       for r in range(len_sizes)]
        lc_ax = fig.add_subplot(gs[1])

        def plot_one_image(ax, data, size, name=None):
            if name == "Difference":
                norm = aviz.ImageNormalize(
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
            plot_one_image(cutout_axes[0][0], template[0].image.array, sizes[0], "Template")
            plot_one_image(cutout_axes[0][1], science[0].image.array, sizes[0], "Science")
            plot_one_image(cutout_axes[0][2], difference[0].image.array, sizes[0], "Difference")
            for i in range(1, len_sizes):
                plot_one_image(cutout_axes[i][0], template[i].image.array, sizes[i], None)
                plot_one_image(cutout_axes[i][1], science[i].image.array, sizes[i], None)
                plot_one_image(cutout_axes[i][2], difference[i].image.array, sizes[i], None)

            self._draw_lightcurve(lc_ax, sources_lc, forced_lc, current_source=source)

            if source is not None and self.config.add_metadata:
                # Place metadata text above the figure top, matching the
                # multi-size layout in the parent class.
                # ``bbox_inches="tight"`` in savefig expands the saved area to
                # include them.
                _annotate_image(fig, source, len_sizes,
                                heights=[1.2, 1.15, 1.1, 1.05, 1.0])

            output = io.BytesIO()
            plt.savefig(output, bbox_inches="tight", format="png")
            output.seek(0)
        finally:
            plt.close(fig)
        return output

    def _draw_lightcurve(self, ax, sources, forced, current_source=None):
        """Draw the lightcurve panel for a diaObject.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            Axes to draw into.
        sources : `pandas.DataFrame` or None
            DiaSources for this diaObject, or None if no APDB is configured.
        forced : `pandas.DataFrame` or None
            DiaForcedSources for this diaObject. Rows whose ``visit`` matches
            one in ``sources`` are suppressed; the rest are drawn with the
            ``lightcurve_marker_forced_only`` marker.
        current_source : `numpy.record`, optional
            The diaSource being cut out. If non-None, a vertical line and
            open ring mark its MJD/psfFlux on the panel.
        """
        if sources is None and forced is None:
            ax.text(0.5, 0.5, "no APDB query configured",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            return

        n_src = 0 if sources is None else len(sources)
        n_forced = 0 if forced is None else len(forced)
        if n_src == 0 and n_forced == 0:
            ax.text(0.5, 0.5, "no lightcurve data",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            return

        # DiaSources are point-like (with the exception of moving objects,
        # which appear only once), so a single visit gives at most one
        # diaSource per diaObject — dedup on ``visit`` is safe.
        if n_forced and n_src:
            forced_only = forced[~forced["visit"].isin(sources["visit"])]
        elif n_forced:
            forced_only = forced
        else:
            forced_only = None

        def _plot_group(group, color, marker, label):
            time_col = plotUtils._time_column(group)
            if "psfFluxErr" in group.columns and group["psfFluxErr"].notna().any():
                ax.errorbar(group[time_col], group["psfFlux"],
                            yerr=group["psfFluxErr"], fmt=marker,
                            color=color, label=label)
            else:
                ax.plot(group[time_col], group["psfFlux"], marker,
                        color=color, label=label)

        if n_src:
            for band, group in sources.groupby("band"):
                color = plotUtils.band_color(band)
                _plot_group(group, color,
                            self.config.lightcurve_marker_source,
                            f"{band} (n={len(group)})")

        if forced_only is not None and len(forced_only):
            # Plot per-band forced points with their band colors, but
            # suppress them from the legend so we only emit one combined
            # entry (in black) for the forced marker regardless of how many
            # bands are present.
            for band, group in forced_only.groupby("band"):
                color = plotUtils.band_color(band)
                _plot_group(group, color,
                            self.config.lightcurve_marker_forced_only,
                            "_nolegend_")
            ax.plot([], [], self.config.lightcurve_marker_forced_only,
                    color="black",
                    label=f"forced (n={len(forced_only)})")

        if self.config.highlight_current_source and current_source is not None:
            try:
                x = float(current_source["midpointMjdTai"])
                y = float(current_source["psfFlux"])
                ax.axvline(x, color="grey", lw=0.5, ls="--")
                ax.plot([x], [y], "o", markerfacecolor="none",
                        markeredgecolor="red", markersize=12,
                        markeredgewidth=1.5)
            except (KeyError, ValueError):
                pass

        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xlabel("MJD (TAI)")
        ax.set_ylabel("psfFlux (nJy)")
        ax.legend(frameon=True, fontsize=7, loc="best")


def _make_apdbQuery(sqlitefile=None, postgres_url=None, namespace=None):
    """Return a query connection to the specified APDB."""
    if sqlitefile is not None:
        return _apdb_mod.ApdbSqliteQuery(sqlitefile)
    if postgres_url is not None and namespace is not None:
        return _apdb_mod.ApdbPostgresQuery(namespace, postgres_url)
    raise RuntimeError("Cannot handle database connection args: "
                       f"sqlitefile={sqlitefile}, postgres_url={postgres_url}, "
                       f"namespace={namespace}")


def build_argparser():
    """Argument parser for the ``plotDiaSourceLightcurve`` command."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="More information is available at https://pipelines.lsst.io.",
    )
    apdbArgs = parser.add_mutually_exclusive_group(required=True)
    apdbArgs.add_argument("--sqlitefile", default=None,
                          help="Path to sqlite APDB file.")
    apdbArgs.add_argument("--namespace", default=None,
                          help="Postgres namespace (schema) to connect to.")
    parser.add_argument(
        "--postgres_url",
        default="rubin@usdf-prompt-processing-dev.slac.stanford.edu/lsst-devl",
        help="Postgres connection path.")
    parser.add_argument("--limit", default=5, type=int,
                        help="Number of sources to load (default=5).")
    parser.add_argument("-C", "--configFile",
                        help="PlotDiaSourceLightcurveConfig file to load.")
    parser.add_argument("--collections", nargs="*",
                        help="Butler collection(s) to load image data from.")
    parser.add_argument("repo", help="Path to butler repository.")
    parser.add_argument("outputPath",
                        help="Path to write images to (under outputPath/images/).")
    parser.add_argument("--reliabilityMin", type=float, default=None,
                        help="Minimum reliability for diaSource selection.")
    parser.add_argument("--reliabilityMax", type=float, default=None,
                        help="Maximum reliability for diaSource selection.")
    return parser


def run_lightcurves(args):
    """Run PlotDiaSourceLightcurveTask from parsed command-line arguments."""
    logging.basicConfig(level=logging.INFO,
                        format="{name} {levelname}: {message}", style="{")

    butler = lsst.daf.butler.Butler(args.repo, collections=args.collections)
    apdb_query = _make_apdbQuery(sqlitefile=args.sqlitefile,
                                 postgres_url=args.postgres_url,
                                 namespace=args.namespace)

    config = PlotDiaSourceLightcurveConfig()
    if args.configFile is not None:
        config.load(os.path.expanduser(args.configFile))
    config.freeze()
    task = PlotDiaSourceLightcurveTask(config=config,
                                       output_path=args.outputPath,
                                       apdb_query=apdb_query)

    data = next(apdb_query.iter_sources(args.limit,
                                        args.reliabilityMin,
                                        args.reliabilityMax))
    sources = task.run(data, butler)
    print(f"Generated {len(sources)} diaSource lightcurve plots to {args.outputPath}.")


def main():
    args = build_argparser().parse_args()
    run_lightcurves(args)

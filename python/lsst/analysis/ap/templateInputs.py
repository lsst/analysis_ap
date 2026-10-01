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

"""Find the images that make up the template at the position of a diaSource.

A template written as an `lsst.images.DifferenceImage` records the coadds it
was built from in its ``templates`` component. Each coadd in turn records
its input images and their weights, either as a legacy
`lsst.afw.image.CoaddInputs` (``coaddInputs`` component) or as an
`lsst.images.cells.CoaddProvenance` (``provenance`` component). The tools
here follow that chain from a diaSource back to the observation times and
fractional weights of the template's input images.
"""

__all__ = ["get_template_inputs", "summarize_template_inputs"]

import logging

import astropy.table
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

import lsst.geom
from lsst.daf.base import DateTime
from lsst.images import Box, Interval

_LOG = logging.getLogger(__name__)

_INPUT_COLUMNS = ["input_visit", "input_detector", "coadd_weight", "mid_mjd_tai", "exposure_time"]
"""Columns of the per-coadd input tables returned by the readers
(`list` [`str`]).
"""


def _empty_inputs():
    """Return a per-coadd input table with no rows."""
    return astropy.table.Table(
        names=["point"] + _INPUT_COLUMNS,
        dtype=[np.int64, np.int64, np.int64, np.float64, np.float64, np.float64],
    )


def _nearest_pixel(value):
    """Return the integer index of the pixel containing a position."""
    return np.floor(np.asarray(value) + 0.5).astype(np.int64)


def _clipped_box(ix, iy, bbox):
    """Return the box enclosing pixel indices, clipped to a bounding box.

    Parameters
    ----------
    ix, iy : `numpy.ndarray`
        Integer pixel indices to enclose.
    bbox : `lsst.images.Box`
        Bounding box of the image to read from.

    Returns
    -------
    box : `lsst.images.Box` or `None`
        Enclosing box, or `None` if no pixel is inside ``bbox``.
    """
    inside = bbox.contains(x=ix, y=iy)
    if not inside.any():
        return None
    return Box(
        y=Interval(iy[inside].min(), iy[inside].max() + 1),
        x=Interval(ix[inside].min(), ix[inside].max() + 1),
    )


class _LegacyCoaddReader:
    """Inputs of a coadd whose provenance is a legacy
    `lsst.afw.image.CoaddInputs`.

    Parameters
    ----------
    ccds : `lsst.afw.table.ExposureCatalog`
        The ``ccds`` catalog of the coadd's ``coaddInputs``.
    butler : `lsst.daf.butler.Butler`, optional
        Butler to read the coadd's variance plane from.
    ref : `lsst.daf.butler.DatasetRef`, optional
        Reference to the coadd.
    """

    def __init__(self, ccds, butler=None, ref=None):
        self._ccds = ccds
        self._butler = butler
        self._ref = ref
        self._mid_mjd = np.full(len(ccds), np.nan)
        self._exposure_time = np.full(len(ccds), np.nan)
        for i, record in enumerate(ccds):
            visit_info = record.getVisitInfo()
            if visit_info is not None:
                self._mid_mjd[i] = visit_info.getDate().get(DateTime.MJD, DateTime.TAI)
                self._exposure_time[i] = visit_info.getExposureTime()

    @classmethod
    def from_butler(cls, butler, ref):
        """Read the provenance of a coadd.

        Parameters
        ----------
        butler : `lsst.daf.butler.Butler`
            Butler to read from.
        ref : `lsst.daf.butler.DatasetRef`
            Reference to the coadd.

        Returns
        -------
        reader : `_LegacyCoaddReader`
            Reader for that coadd.
        """
        coadd_inputs = butler.get(ref.makeComponentRef("coaddInputs"))
        return cls(coadd_inputs.ccds, butler=butler, ref=ref)

    def find_inputs(self, ra, dec):
        """Find the input images that cover sky positions.

        Parameters
        ----------
        ra, dec : `numpy.ndarray`
            Sky positions, in degrees.

        Returns
        -------
        inputs : `astropy.table.Table`
            One row per (position, input image), with ``point`` giving the
            index of the position.

        Notes
        -----
        An input covers a position if the position is inside both the bounding
        box and the valid polygon of that input, the same rule
        `lsst.meas.algorithms.CoaddPsf` uses.
        """
        points = [lsst.geom.SpherePoint(r, d, lsst.geom.degrees) for r, d in zip(ra, dec)]
        rows = []
        for i, record in enumerate(self._ccds):
            bbox = lsst.geom.Box2D(record.getBBox())
            polygon = record.getValidPolygon()
            for point, pixel in enumerate(record.getWcs().skyToPixel(points)):
                if bbox.contains(pixel) and (polygon is None or polygon.contains(pixel)):
                    rows.append((point, record["visit"], record["ccd"], record["weight"],
                                 self._mid_mjd[i], self._exposure_time[i]))
        if not rows:
            return _empty_inputs()
        return astropy.table.Table(rows=rows, names=_empty_inputs().colnames)

    def variance_at(self, ra, dec):
        """Read the coadd variance at sky positions.

        Parameters
        ----------
        ra, dec : `numpy.ndarray`
            Sky positions, in degrees.

        Returns
        -------
        variance : `numpy.ndarray`
            Variance of the nearest coadd pixel; ``nan`` outside the coadd.
        """
        wcs = self._butler.get(self._ref.makeComponentRef("wcs"))
        points = [lsst.geom.SpherePoint(r, d, lsst.geom.degrees) for r, d in zip(ra, dec)]
        pixels = wcs.skyToPixel(points)
        ix = _nearest_pixel([p.getX() for p in pixels])
        iy = _nearest_pixel([p.getY() for p in pixels])
        coadd_bbox = Box.from_legacy(self._butler.get(self._ref.makeComponentRef("bbox")))
        box = _clipped_box(ix, iy, coadd_bbox)
        result = np.full(len(ix), np.nan)
        if box is None:
            return result
        variance = self._butler.get(self._ref.makeComponentRef("variance"),
                                    parameters={"bbox": box.to_legacy()})
        inside = box.contains(x=ix, y=iy)
        result[inside] = variance.array[iy[inside] - box.y.start, ix[inside] - box.x.start]
        return result


class _CellCoaddReader:
    """Inputs of a coadd whose provenance is an
    `lsst.images.cells.CoaddProvenance`.

    Parameters
    ----------
    provenance : `lsst.images.cells.CoaddProvenance`
        Provenance of the coadd.
    sky_projection : `lsst.images.SkyProjection`
        Mapping from the coadd's pixels to the sky.
    grid : `lsst.images.CellGrid`
        Cell grid of the coadd's patch.
    times : `dict` [`tuple` [`str`, `int`], `tuple` [`float`, `float`]]
        Mid-exposure time (MJD, TAI) and exposure time (s), keyed by
        ``(instrument, visit)``.
    butler : `lsst.daf.butler.Butler`, optional
        Butler to read the coadd's variance plane from.
    ref : `lsst.daf.butler.DatasetRef`, optional
        Reference to the coadd.
    """

    def __init__(self, provenance, sky_projection, grid, times, butler=None, ref=None):
        self._sky_projection = sky_projection
        self._grid = grid
        self._butler = butler
        self._ref = ref
        self._polygons = {
            (str(row["instrument"]), int(row["visit"]), int(row["detector"])): row["polygon"]
            for row in provenance.inputs
        }
        self._contributions = provenance.contributions
        self._rows_by_cell = {}
        for n, row in enumerate(self._contributions):
            self._rows_by_cell.setdefault((int(row["cell_i"]), int(row["cell_j"])), []).append(n)
        self._times = times

    @classmethod
    def from_butler(cls, butler, ref):
        """Read the provenance of a coadd and the times of its inputs.

        Parameters
        ----------
        butler : `lsst.daf.butler.Butler`
            Butler to read from.
        ref : `lsst.daf.butler.DatasetRef`
            Reference to the coadd.

        Returns
        -------
        reader : `_CellCoaddReader`
            Reader for that coadd.
        """
        parts = butler.get(ref.makeComponentRef("components"),
                           parameters={"components": ["provenance", "sky_projection", "psf"]})
        provenance = parts["provenance"]
        times = {}
        for instrument in np.unique(provenance.inputs["instrument"]):
            visits = provenance.inputs["visit"][provenance.inputs["instrument"] == instrument]
            records = butler.query_dimension_records(
                "visit", instrument=str(instrument), where="visit IN (:visits)",
                bind={"visits": [int(v) for v in np.unique(visits)]}, limit=None,
            )
            for record in records:
                timespan = record.timespan
                mid = timespan.begin + (timespan.end - timespan.begin)/2
                times[str(instrument), record.id] = (mid.tai.mjd, record.exposure_time)
        return cls(provenance, parts["sky_projection"], parts["psf"].bounds.grid, times,
                   butler=butler, ref=ref)

    def _pixels(self, ra, dec):
        """Return the coadd pixel positions of sky positions."""
        xy = self._sky_projection.sky_to_pixel(SkyCoord(ra*u.deg, dec*u.deg))
        return np.atleast_1d(xy.x), np.atleast_1d(xy.y)

    def find_inputs(self, ra, dec):
        """Find the input images that cover sky positions.

        Parameters
        ----------
        ra, dec : `numpy.ndarray`
            Sky positions, in degrees.

        Returns
        -------
        inputs : `astropy.table.Table`
            One row per (position, input image), with ``point`` giving the
            index of the position.

        Notes
        -----
        An input covers a position if it contributes to the cell containing
        the position and its overlap polygon contains the position. The
        weight is the one recorded for that cell.
        """
        x, y = self._pixels(ra, dec)
        rows = []
        for point, (px, py) in enumerate(zip(x, y)):
            cell = self._grid.index_of(y=int(_nearest_pixel(py)), x=int(_nearest_pixel(px)))
            for n in self._rows_by_cell.get((cell.i, cell.j), []):
                row = self._contributions[n]
                instrument = str(row["instrument"])
                key = (instrument, int(row["visit"]), int(row["detector"]))
                if not self._polygons[key].contains(x=px, y=py):
                    continue
                mid_mjd, exposure_time = self._times.get(key[:2], (np.nan, np.nan))
                rows.append((point, key[1], key[2], row["weight"], mid_mjd, exposure_time))
        if not rows:
            return _empty_inputs()
        return astropy.table.Table(rows=rows, names=_empty_inputs().colnames)

    def variance_at(self, ra, dec):
        """Read the coadd variance at sky positions.

        Parameters
        ----------
        ra, dec : `numpy.ndarray`
            Sky positions, in degrees.

        Returns
        -------
        variance : `numpy.ndarray`
            Variance of the nearest coadd pixel; ``nan`` outside the coadd.
        """
        x, y = self._pixels(ra, dec)
        ix = _nearest_pixel(x)
        iy = _nearest_pixel(y)
        box = _clipped_box(ix, iy, self._grid.bbox)
        result = np.full(len(ix), np.nan)
        if box is None:
            return result
        variance = self._butler.get(self._ref.makeComponentRef("variance"), parameters={"bbox": box})
        inside = box.contains(x=ix, y=iy)
        result[inside] = variance.array[iy[inside] - box.y.start, ix[inside] - box.x.start]
        return result


_READERS = {"coaddInputs": _LegacyCoaddReader, "provenance": _CellCoaddReader}
"""Coadd reader for each provenance component, in order of preference
(`dict` [`str`, `type`]).
"""


def _get_reader(butler, info, readers):
    """Return the reader for one template coadd, reading it if needed.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler to read from.
    info : `lsst.images.DifferenceImageTemplateInfo`
        Record of the coadd.
    readers : `dict`
        Readers already made, keyed by dataset ID; updated in place.

    Returns
    -------
    reader : `_LegacyCoaddReader` or `_CellCoaddReader`
        Reader for the coadd.

    Raises
    ------
    LookupError
        Raised if the butler has no dataset with the coadd's ID.
    TypeError
        Raised if the coadd's storage class has no known provenance
        component.
    """
    if info.dataset_id in readers:
        return readers[info.dataset_id]
    ref = butler.get_dataset(info.dataset_id)
    if ref is None:
        raise LookupError(f"Template coadd tract={info.tract}, patch={info.patch} "
                          f"(id={info.dataset_id}, run={info.dataset_run!r}) is not in this butler.")
    components = ref.datasetType.storageClass.allComponents()
    for component, reader_type in _READERS.items():
        if component in components:
            readers[info.dataset_id] = reader_type.from_butler(butler, ref)
            return readers[info.dataset_id]
    raise TypeError(f"Coadd {ref} has storage class {ref.datasetType.storageClass_name}, which has none "
                    f"of the provenance components {list(_READERS)}.")


def _combine_weights(coadds, variances):
    """Compute the fraction of the template flux at one position that came
    from each input image.

    Parameters
    ----------
    coadds : `list` [`tuple`]
        One ``(key, tract, inputs)`` entry per coadd whose template bounds
        contain the position, where ``inputs`` is that coadd's input table
        for the position.
    variances : `dict`
        Coadd variance at the position, keyed by ``key``. Needed only when
        there is more than one coadd.

    Returns
    -------
    fractions : `dict`
        Weight to apply to each coadd's ``coadd_weight`` column, keyed by
        ``key``. Coadds that contribute nothing are omitted.

    Notes
    -----
    This follows `lsst.ip.diffim.GetTemplateTask`: the patches of each tract
    are averaged with per-pixel weights ``variance**-0.5``, and the tracts
    are then averaged the same way using the averaged variance of each tract.
    Within one coadd, the fraction from an input is its coadd weight over the
    sum of the weights of the inputs covering the position.
    """
    coadds = [(key, tract, inputs) for key, tract, inputs in coadds if inputs["coadd_weight"].sum() > 0]
    if len(coadds) == 1:
        key, _, inputs = coadds[0]
        return {key: 1.0/inputs["coadd_weight"].sum()}
    patch_weights = {}
    tract_variances = {}
    for key, tract, inputs in coadds:
        variance = variances.get(key, np.nan)
        if np.isfinite(variance) and variance > 0:
            patch_weights.setdefault(tract, {})[key] = variance**-0.5
    for tract, weights in patch_weights.items():
        total = sum(weights.values())
        for key in weights:
            weights[key] /= total
        tract_variances[tract] = sum(weights[key]*variances[key] for key in weights)
    tract_weights = {tract: variance**-0.5 for tract, variance in tract_variances.items()}
    tract_total = sum(tract_weights.values())
    fractions = {}
    for key, tract, inputs in coadds:
        if key in patch_weights.get(tract, {}):
            fractions[key] = (tract_weights[tract]/tract_total*patch_weights[tract][key]
                              / inputs["coadd_weight"].sum())
    return fractions


def _template_inputs_for_detector(butler, data_id, sources, template_dataset, readers):
    """Find the template inputs for the diaSources on one detector.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler to read from.
    data_id : `dict`
        Data ID of the detector.
    sources : `astropy.table.Table`
        DiaSources on the detector.
    template_dataset : `str`
        Name of the template dataset type.
    readers : `dict`
        Coadd readers already made, keyed by dataset ID; updated in place.

    Returns
    -------
    inputs : `list` [`astropy.table.Table`]
        Input tables, one per contributing coadd.
    """
    templates = butler.get(f"{template_dataset}.templates", data_id)
    ra = np.asarray(sources["ra"], dtype=float)
    dec = np.asarray(sources["dec"], dtype=float)
    x = np.asarray(sources["x"], dtype=float)
    y = np.asarray(sources["y"], dtype=float)
    per_coadd = []
    for info in templates:
        contains = np.atleast_1d(info.bounds.contains(x=x, y=y))
        if not contains.any():
            continue
        indices = np.flatnonzero(contains)
        reader = _get_reader(butler, info, readers)
        inputs = reader.find_inputs(ra[indices], dec[indices])
        inputs["point"] = indices[inputs["point"]]
        per_coadd.append((info, reader, indices, inputs))
    n_coadds = np.zeros(len(sources), dtype=int)
    for _, _, indices, _ in per_coadd:
        n_coadds[indices] += 1
    variances = {}
    for info, reader, indices, _ in per_coadd:
        overlap = indices[n_coadds[indices] > 1]
        if len(overlap):
            for point, variance in zip(overlap, reader.variance_at(ra[overlap], dec[overlap])):
                variances[point, info.dataset_id] = variance
    results = []
    for point in range(len(sources)):
        coadds = []
        infos = {}
        for info, _, _, inputs in per_coadd:
            mine = inputs[inputs["point"] == point]
            if len(mine):
                coadds.append(((point, info.dataset_id), info.tract, mine))
                infos[point, info.dataset_id] = info
        fractions = _combine_weights(coadds, variances) if coadds else {}
        if not fractions:
            _LOG.warning("No template coadd input contributes to diaSource %s.",
                         sources["diaSourceId"][point])
            continue
        for key, _, inputs in coadds:
            if key not in fractions:
                continue
            table = inputs.copy()
            table["tract"] = infos[key].tract
            table["patch"] = infos[key].patch
            table["weight"] = table["coadd_weight"]*fractions[key]
            results.append(table)
    return results


def get_template_inputs(butler, dia_sources, *, instrument=None, template_dataset="template_detector"):
    """Find the observation times and weights of the images that make up
    the template at the position of each diaSource.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler with the templates and the coadds they were built from.
        Coadds are found by dataset ID, so their collections need not be
        searched.
    dia_sources : `astropy.table.Table` or `pandas.DataFrame`
        DiaSources, with columns ``diaSourceId``, ``visit``, ``detector``,
        ``ra``, ``dec`` (degrees) and ``x``, ``y`` (science image pixels).
    instrument : `str`, optional
        Instrument name, if the butler has no default.
    template_dataset : `str`, optional
        Template dataset type; must be an `lsst.images.DifferenceImage`.

    Returns
    -------
    inputs : `astropy.table.Table`
        One row per (diaSource, template coadd, input image), with columns:

        ``diaSourceId``, ``visit``, ``detector``
            The diaSource and its science image.
        ``tract``, ``patch``
            Template coadd the input contributed through.
        ``input_visit``, ``input_detector``
            The input image.
        ``mid_mjd_tai``
            Mid-exposure time of the input (MJD, TAI); ``nan`` if unknown.
        ``exposure_time``
            Exposure time of the input (s); ``nan`` if unknown.
        ``coadd_weight``
            Weight of the input as recorded by the coadd.
        ``weight``
            Fraction of the template flux at the diaSource that came from
            this input through this coadd. These sum to 1 for each
            diaSource; an input in several overlapping coadds has one row
            for each.

    Raises
    ------
    TypeError
        Raised if ``template_dataset`` has no ``templates`` component (a
        legacy Exposure), or if a coadd has no known provenance component.
    LookupError
        Raised if a template coadd is missing from the butler.

    Notes
    -----
    The weights are evaluated at the diaSource position only. Warping the
    template and PSF matching both mix in nearby template pixels, which can
    come from other coadds or inputs; that is not modeled. Per-pixel input
    rejection (masked pixels, or artifacts clipped by
    `lsst.drp.tasks.assemble_coadd.CompareWarpAssembleCoaddTask`) is not
    recorded in coadd provenance, so a rejected input is still reported.
    """
    dataset_type = butler.get_dataset_type(template_dataset)
    if "templates" not in dataset_type.storageClass.allComponents():
        raise TypeError(f"{template_dataset!r} has storage class {dataset_type.storageClass_name}, which "
                        "has no 'templates' component; templates written as legacy Exposures are not "
                        "supported.")
    if isinstance(dia_sources, pd.DataFrame):
        sources = astropy.table.Table.from_pandas(dia_sources)
    else:
        sources = astropy.table.Table(dia_sources)
    readers = {}
    results = []
    pairs = np.unique(np.column_stack([sources["visit"], sources["detector"]]), axis=0)
    for visit, detector in pairs:
        selected = sources[(sources["visit"] == visit) & (sources["detector"] == detector)]
        data_id = {"visit": int(visit), "detector": int(detector)}
        if instrument is not None:
            data_id["instrument"] = instrument
        for table in _template_inputs_for_detector(butler, data_id, selected, template_dataset, readers):
            table["diaSourceId"] = selected["diaSourceId"][table["point"]]
            table["visit"] = visit
            table["detector"] = detector
            results.append(table)
    columns = ["diaSourceId", "visit", "detector", "tract", "patch"] + _INPUT_COLUMNS + ["weight"]
    if not results:
        empty = _empty_inputs()
        for name in ["diaSourceId", "visit", "detector", "tract", "patch"]:
            empty[name] = np.zeros(0, dtype=np.int64)
        empty["weight"] = np.zeros(0)
        return empty[columns]
    return astropy.table.vstack(results)[columns]


def summarize_template_inputs(inputs):
    """Summarize the template inputs of each diaSource.

    Parameters
    ----------
    inputs : `astropy.table.Table`
        Output of `get_template_inputs`.

    Returns
    -------
    summary : `astropy.table.Table`
        One row per diaSource, with columns ``diaSourceId``, ``n_inputs``
        (distinct input images), ``mean_mjd_tai`` (weighted mean
        mid-exposure time), ``min_mjd_tai`` and ``max_mjd_tai``. Inputs with
        unknown times are left out of the time columns.
    """
    rows = []
    for group in inputs.group_by("diaSourceId").groups:
        known = np.isfinite(group["mid_mjd_tai"])
        times = group["mid_mjd_tai"][known]
        weights = group["weight"][known]
        n_inputs = len(np.unique(np.column_stack([group["input_visit"], group["input_detector"]]), axis=0))
        if len(times):
            row = (np.average(times, weights=weights) if weights.sum() > 0 else np.nan,
                   times.min(), times.max())
        else:
            row = (np.nan, np.nan, np.nan)
        rows.append((group["diaSourceId"][0], n_inputs) + row)
    return astropy.table.Table(
        rows=rows,
        names=["diaSourceId", "n_inputs", "mean_mjd_tai", "min_mjd_tai", "max_mjd_tai"],
        dtype=[np.int64, np.int64, np.float64, np.float64, np.float64],
    )

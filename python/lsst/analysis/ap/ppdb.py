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

"""Load PPDB catalogs (DiaObjects, DiaSources, DiaForcedSources) through the
Rubin Science Platform TAP service.

These tools deliberately use the scientist-facing TAP interface (ADQL over
``pyvo``) rather than a direct database connection, so that exercising them
also exercises the interface scientists will use once the PPDB goes live.

The prototype PPDB currently lives at the ``data-int`` RSP environment
(``https://data-int.lsst.cloud/api/ppdbtap``). Tooling running elsewhere
(e.g. the USDF RSP) reaches it cross-environment with a ``data-int`` RSP
token carrying the ``read:tap`` scope, supplied via the ``RSP_TOKEN``
environment variable.

The ``DiaObject`` table is versioned: a single object accumulates multiple
rows over time and only the row with ``validityEndMjdTai IS NULL`` is the
current version. Every DiaObject query here applies that filter by default
so callers never accidentally retrieve (or double-count) stale versions.
``DiaSource`` and ``DiaForcedSource`` are append-only and are not versioned.

The production PPDB will allow cone searches only on ``DiaObject``. Sources
and forced sources for a region must therefore be loaded object-first:
cone-search DiaObjects, then load the sources of those objects. The
``load_sources_for_region`` / ``load_forced_sources_for_region`` methods do
this for you and are the encouraged entry points. ``load_sources`` /
``load_forced_sources`` require a ``diaObjectId`` (scalar or iterable) and
take no spatial argument. Direct, object-unrestricted access to the source
tables is available only through the explicit, prototype-only
``load_sources_by_cone`` / ``load_forced_sources_by_cone`` (spatial) and
``load_sources_by_time_window`` / ``load_forced_sources_by_time_window``
(temporal) methods, which warn when used.

All loaders return `astropy.table.Table` objects.
"""

__all__ = [
    "PpdbTap",
    "DiaObjectLightCurve",
    "region_from_exposure",
    "DEFAULT_PPDB_TAP_URL",
    "DEFAULT_PADDING_ARCSEC",
    "DEFAULT_ROW_LIMIT",
]

import dataclasses
import logging
import numbers
import os
import time

import astropy.table
import pyvo
import requests

import lsst.geom

_log = logging.getLogger(__name__)

# TAP endpoint of the prototype PPDB (the ``data-int`` RSP environment).
DEFAULT_PPDB_TAP_URL = "https://data-int.lsst.cloud/api/ppdbtap"

# Default padding (arcseconds) loaded around an exposure footprint.
DEFAULT_PADDING_ARCSEC = 5.0

# Default cap on rows returned by a single query, to guard against
# accidentally pulling the whole (multi-million row) PPDB. Pass ``limit=None``
# to disable.
DEFAULT_ROW_LIMIT = 100000

# Valid LSST band names; used to validate band filters before they are
# interpolated into an ADQL string.
_VALID_BANDS = frozenset("ugrizy")


def region_from_exposure(exposure, padding=DEFAULT_PADDING_ARCSEC):
    """Compute a cone (ra, dec, radius) covering an exposure's sky footprint.

    The cone is the smallest circle centered on the exposure's bounding-box
    center that contains all four corners, expanded by ``padding``
    arcseconds. A circumscribing circle (rather than the exact polygon) keeps
    the resulting ADQL robust: ``CIRCLE`` is universally supported by TAP
    services, whereas ``POLYGON`` support and winding conventions vary.

    Parameters
    ----------
    exposure : `lsst.afw.image.Exposure`
        Exposure whose WCS and bounding box define the region.
    padding : `float`, optional
        Margin to add around the footprint, in arcseconds.

    Returns
    -------
    ra : `float`
        Cone center right ascension, in degrees (ICRS).
    dec : `float`
        Cone center declination, in degrees (ICRS).
    radius : `float`
        Cone radius, in degrees.

    Raises
    ------
    ValueError
        If the exposure has no WCS, an empty bounding box, or negative
        padding.
    """
    if padding < 0:
        raise ValueError(f"padding must be non-negative, got {padding}")
    wcs = exposure.getWcs()
    if wcs is None:
        raise ValueError("Exposure has no WCS; cannot compute a sky region.")
    bbox = exposure.getBBox()
    if bbox.isEmpty():
        raise ValueError("Exposure has an empty bounding box.")

    center = wcs.pixelToSky(bbox.getCenter())
    corners = [wcs.pixelToSky(lsst.geom.Point2D(corner))
               for corner in bbox.getCorners()]
    radius = max(center.separation(corner) for corner in corners)
    radius = radius + padding * lsst.geom.arcseconds
    return (center.getRa().asDegrees(),
            center.getDec().asDegrees(),
            radius.asDegrees())


@dataclasses.dataclass
class DiaObjectLightCurve:
    """The full PPDB record for a single diaObject.

    Attributes
    ----------
    diaObjectId : `int`
        Identifier of the diaObject.
    dia_object : `astropy.table.Row`
        The current (latest-version) DiaObject row.
    dia_sources : `astropy.table.Table`
        All DiaSources associated with the object, time-ordered.
    dia_forced_sources : `astropy.table.Table`
        All DiaForcedSources for the object, time-ordered.
    """

    diaObjectId: int
    dia_object: "astropy.table.Row"
    dia_sources: astropy.table.Table
    dia_forced_sources: astropy.table.Table

    @property
    def n_sources(self):
        """Number of associated DiaSources (`int`)."""
        return len(self.dia_sources)

    @property
    def n_forced_sources(self):
        """Number of associated DiaForcedSources (`int`)."""
        return len(self.dia_forced_sources)

    def __repr__(self):
        return (f"DiaObjectLightCurve(diaObjectId={self.diaObjectId}, "
                f"n_sources={self.n_sources}, "
                f"n_forced_sources={self.n_forced_sources})")


class PpdbTap:
    """Load PPDB catalogs through the RSP TAP service as astropy Tables.

    Parameters
    ----------
    url : `str`, optional
        TAP endpoint to query. Defaults to the prototype PPDB at ``data-int``.
    token : `str`, optional
        RSP token with the ``read:tap`` scope. If not given (and ``service``
        is also not given), the ``RSP_TOKEN`` environment variable is used.
    service : `pyvo.dal.TAPService`, optional
        Pre-built TAP service. When supplied, ``url`` and ``token`` are
        ignored. Intended for use with unit tests.
    log : `logging.Logger`, optional
        Logger to use; defaults to the module logger.

    Notes
    -----
    The token is read only from the environment or the constructor argument
    and is never logged.
    """

    def __init__(self, url=DEFAULT_PPDB_TAP_URL, *, token=None,
                 service=None, log=None):
        self.url = url
        self.log = log if log is not None else _log
        if service is not None:
            self._service = service
            return
        if token is None:
            token = os.environ.get("RSP_TOKEN")
        if not token:
            raise RuntimeError(
                "No RSP token available for the PPDB TAP service. Generate "
                "a token with the 'read:tap' scope on data-int "
                "(https://data-int.lsst.cloud) and set it in the RSP_TOKEN "
                "environment variable, or pass token=...")
        session = requests.Session()
        session.headers["Authorization"] = f"Bearer {token}"
        self._service = pyvo.dal.TAPService(url, session=session)

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------
    def run_query(self, adql, *, maxrec=None):
        """Run an ADQL query and return the result as an astropy Table.

        Parameters
        ----------
        adql : `str`
            The ADQL query to execute.
        maxrec : `int`, optional
            Server-side cap on the number of records returned.

        Returns
        -------
        table : `astropy.table.Table`
            The query result.
        """
        self.log.debug("Executing ADQL query: %s", adql)
        start = time.perf_counter()
        results = self._service.run_async(adql, maxrec=maxrec)
        table = results.to_table()
        elapsed = time.perf_counter() - start
        self.log.info("Query returned %d rows in %.3f s", len(table), elapsed)
        return table

    # ------------------------------------------------------------------
    # ADQL construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _select_clause(columns, limit):
        """Build the ``SELECT [TOP n] cols`` prefix of a query."""
        cols = "*" if not columns else ", ".join(columns)
        top = f"TOP {int(limit)} " if limit is not None else ""
        return f"SELECT {top}{cols}"

    @staticmethod
    def _cone_clause(ra, dec, radius):
        """Build an ADQL cone-search predicate (all angles in degrees)."""
        return (f"CONTAINS(POINT('ICRS', ra, dec), "
                f"CIRCLE('ICRS', {float(ra)}, {float(dec)}, "
                f"{float(radius)})) = 1")

    def _resolve_cone(self, ra, dec, radius, exposure, padding):
        """Resolve spatial arguments into a (ra, dec, radius) cone or None.

        Either an ``exposure`` or an explicit ``ra``/``dec``/``radius`` triple
        may be given, but not both. Returns None when no spatial constraint
        was requested. Explicit ``radius`` is in degrees; ``padding`` (used
        only with ``exposure``) is in arcseconds.
        """
        if exposure is not None:
            if any(v is not None for v in (ra, dec, radius)):
                raise ValueError("Specify either exposure= or ra/dec/radius, "
                                 "not both.")
            return region_from_exposure(exposure, padding=padding)
        if ra is None and dec is None and radius is None:
            return None
        if None in (ra, dec, radius):
            raise ValueError("ra, dec, and radius must all be provided for a "
                             "cone search.")
        return float(ra), float(dec), float(radius)

    @staticmethod
    def _band_clause(bands):
        """Build an ADQL ``band IN (...)`` predicate, validating each band."""
        quoted = []
        for band in bands:
            if band not in _VALID_BANDS:
                raise ValueError(f"Unknown band {band!r}; expected one of "
                                 f"{sorted(_VALID_BANDS)}.")
            quoted.append(f"'{band}'")
        return f"band IN ({', '.join(quoted)})"

    def _finish(self, table, limit, table_name):
        """Warn on probable truncation and return the table unchanged."""
        if limit is not None and len(table) == limit:
            self.log.warning(
                "%s query returned exactly the row limit (%d); results are "
                "likely truncated. Narrow the query or raise limit.",
                table_name, limit)
        return table

    # ------------------------------------------------------------------
    # DiaObject loaders
    # ------------------------------------------------------------------
    def load_objects(self, *, ra=None, dec=None, radius=None, exposure=None,
                     padding=DEFAULT_PADDING_ARCSEC, columns=None,
                     limit=DEFAULT_ROW_LIMIT, latest=True):
        """Load DiaObjects, optionally within a spatial region.

        Parameters
        ----------
        ra, dec, radius : `float`, optional
            Cone-search center and radius, in degrees. All three must be
            given together; mutually exclusive with ``exposure``.
        exposure : `lsst.afw.image.Exposure`, optional
            Load objects covering this exposure's footprint (see
            `region_from_exposure`). Mutually exclusive with
            ``ra``/``dec``/``radius``.
        padding : `float`, optional
            Margin around the exposure footprint, in arcseconds (only used
            with ``exposure``).
        columns : `list` [`str`], optional
            Columns to select; defaults to all columns.
        limit : `int`, optional
            Maximum number of rows to return; None means no limit.
        latest : `bool`, optional
            If True (default), return only the current version of each object
            (``validityEndMjdTai IS NULL``). Setting this False returns every
            historical version and is rarely what you want.

        Returns
        -------
        objects : `astropy.table.Table`
            The matching DiaObjects.
        """
        cone = self._resolve_cone(ra, dec, radius, exposure, padding)
        clauses = []
        if latest:
            clauses.append("validityEndMjdTai IS NULL")
        if cone is not None:
            clauses.append(self._cone_clause(*cone))
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        adql = (f"{self._select_clause(columns, limit)} FROM ppdb.DiaObject"
                f"{where} ORDER BY diaObjectId")
        table = self._finish(self.run_query(adql), limit, "DiaObject")
        if latest and len(table) and "diaObjectId" in table.colnames:
            n_unique = len(set(table["diaObjectId"].tolist()))
            if n_unique != len(table):
                self.log.warning(
                    "DiaObject query returned %d rows but only %d unique "
                    "diaObjectIds despite validityEndMjdTai IS NULL; the PPDB "
                    "may contain duplicate current versions.",
                    len(table), n_unique)
        return table

    def load_object(self, diaObjectId, *, columns=None):
        """Load the current version of a single DiaObject.

        Parameters
        ----------
        diaObjectId : `int`
            Identifier of the object to load.
        columns : `list` [`str`], optional
            Columns to select; defaults to all columns.

        Returns
        -------
        object : `astropy.table.Row`
            The current DiaObject row.

        Raises
        ------
        RuntimeError
            If no current version of the object exists.
        """
        adql = (f"{self._select_clause(columns, None)} FROM ppdb.DiaObject "
                f"WHERE validityEndMjdTai IS NULL AND "
                f"diaObjectId = {int(diaObjectId)}")
        table = self.run_query(adql)
        if len(table) == 0:
            raise RuntimeError(
                f"diaObjectId={diaObjectId} not found in ppdb.DiaObject")
        if len(table) > 1:
            self.log.warning(
                "diaObjectId=%s has %d current versions (validityEndMjdTai "
                "IS NULL); returning the first.", diaObjectId, len(table))
        return table[0]

    # ------------------------------------------------------------------
    # DiaSource / DiaForcedSource loaders
    #
    # The production PPDB will permit cone searches only on DiaObject. The
    # encouraged way to retrieve sources for a region is therefore object-
    # first: cone-search DiaObjects (`load_*_for_region`), then load their
    # sources by diaObjectId. `load_sources`/`load_forced_sources` require a
    # diaObjectId and take no spatial argument. Direct, object-unrestricted
    # source-table access is confined to the explicit, prototype-only
    # `load_*_by_cone` and `load_*_by_time_window` methods.
    # ------------------------------------------------------------------
    def _source_filter_clauses(self, bands, mjd_begin, mjd_end):
        """Build the time/band filter clauses shared by the source loaders."""
        clauses = []
        if mjd_begin is not None:
            clauses.append(f"midpointMjdTai >= {float(mjd_begin)}")
        if mjd_end is not None:
            clauses.append(f"midpointMjdTai < {float(mjd_end)}")
        if bands:
            clauses.append(self._band_clause(bands))
        return clauses

    def _run_source_query(self, table_name, clauses, columns, limit):
        """Run a single time-ordered SELECT against a source table."""
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        adql = (f"{self._select_clause(columns, limit)} FROM ppdb.{table_name}"
                f"{where} ORDER BY midpointMjdTai")
        return self._finish(self.run_query(adql), limit, table_name)

    def _load_source_like(self, table_name, diaObjectId, extra_clauses,
                          columns, limit, id_chunk_size):
        """Dispatch a source load by one id or many ids.

        ``diaObjectId`` is required: source tables are loaded by object. An
        iterable (including the empty list a region search may produce) takes
        the chunked ``IN`` path; a scalar takes a single ``=`` query.
        """
        if diaObjectId is None:
            raise ValueError(
                "diaObjectId is required; DiaSources and DiaForcedSources are "
                "loaded by object. Use load_*_for_region for a spatial "
                "region, or the prototype-only load_*_by_cone / "
                "load_*_by_time_window methods for direct source-table access.")
        if isinstance(diaObjectId, numbers.Integral):
            clauses = [f"diaObjectId = {int(diaObjectId)}"] + list(extra_clauses)
            return self._run_source_query(table_name, clauses, columns, limit)
        return self._load_sources_for_ids(table_name, diaObjectId,
                                          extra_clauses, columns, limit,
                                          id_chunk_size)

    def _load_sources_for_ids(self, table_name, ids, extra_clauses, columns,
                              limit, id_chunk_size):
        """Load sources for many diaObjectIds via chunked ``IN`` queries.

        A cone search can return more diaObjectIds than fit in a single
        ``diaObjectId IN (...)`` clause, so the ids are split into chunks
        of ``id_chunk_size`` and the per-chunk results are concatenated and
        re-sorted by time.
        ``limit`` caps the total rows returned across all chunks.
        """
        if id_chunk_size < 1:
            raise ValueError("id_chunk_size must be >= 1")
        ids = list(dict.fromkeys(int(i) for i in ids))
        if not ids or (limit is not None and limit <= 0):
            # Empty result, but issue a query so columns/dtypes are populated.
            return self._run_source_query(table_name, ["1 = 0"], columns, None)

        pages = []
        total = 0
        truncated = False
        for start in range(0, len(ids), id_chunk_size):
            if limit is not None and total >= limit:
                truncated = True
                break
            chunk = ids[start:start + id_chunk_size]
            remaining = None if limit is None else limit - total
            id_list = ", ".join(str(i) for i in chunk)
            clauses = [f"diaObjectId IN ({id_list})"] + list(extra_clauses)
            where = " WHERE " + " AND ".join(clauses)
            adql = (f"{self._select_clause(columns, remaining)} "
                    f"FROM ppdb.{table_name}{where} ORDER BY midpointMjdTai")
            page = self.run_query(adql)
            total += len(page)
            pages.append(page)

        table = (pages[0] if len(pages) == 1
                 else astropy.table.vstack(pages, metadata_conflicts="silent"))
        if len(table) and "midpointMjdTai" in table.colnames:
            table.sort("midpointMjdTai")
        if limit is not None and (truncated or total == limit):
            self.log.warning(
                "%s query reached the row limit (%d) while loading sources for "
                "%d objects; results are likely truncated. Raise limit or "
                "narrow the query.", table_name, limit, len(ids))
        return table

    def _direct_source_query(self, table_name, region_method, kind, clauses,
                             columns, limit):
        """Warn, then run a prototype-only direct source-table query.

        ``load_sources``/``load_forced_sources`` require a diaObjectId; these
        direct (object-unrestricted) source-table searches will not exist in
        the production PPDB, so callers reach them only through the explicit
        ``load_*_by_cone`` / ``load_*_by_time_window`` methods, which route
        here and emit this warning.
        """
        self.log.warning(
            "Direct %s %s search is a prototype-only capability and will not "
            "be available in the production PPDB; prefer %s().",
            table_name, kind, region_method)
        return self._run_source_query(table_name, clauses, columns, limit)

    def _load_by_cone(self, table_name, region_method, *, ra, dec, radius,
                      exposure, padding, bands, mjd_begin, mjd_end, columns,
                      limit):
        """Direct cone search on a source table (prototype-only)."""
        cone = self._resolve_cone(ra, dec, radius, exposure, padding)
        if cone is None:
            raise ValueError("A cone search requires ra/dec/radius or an "
                             "exposure.")
        clauses = [self._cone_clause(*cone)]
        clauses += self._source_filter_clauses(bands, mjd_begin, mjd_end)
        return self._direct_source_query(table_name, region_method, "cone",
                                         clauses, columns, limit)

    def _load_by_time_window(self, table_name, region_method, *, mjd_begin,
                             mjd_end, bands, columns, limit):
        """Direct time-window scan of a source table (prototype-only)."""
        if mjd_begin is None and mjd_end is None:
            raise ValueError("A time-window search requires mjd_begin and/or "
                             "mjd_end.")
        clauses = self._source_filter_clauses(bands, mjd_begin, mjd_end)
        return self._direct_source_query(table_name, region_method,
                                         "time-window", clauses, columns,
                                         limit)

    def load_sources(self, *, diaObjectId, mjd_begin=None, mjd_end=None,
                     bands=None, columns=None, limit=DEFAULT_ROW_LIMIT,
                     id_chunk_size=1000):
        """Load DiaSources for a given object (or objects), by time and band.

        DiaSources are loaded by object: ``diaObjectId`` is required. This
        loader has no spatial argument, because the production PPDB does not
        permit searches on the source tables. To load sources for a region,
        use `load_sources_for_region` (object-first); for direct, object-
        unrestricted prototype-only access, use `load_sources_by_cone` or
        `load_sources_by_time_window`.

        Parameters
        ----------
        diaObjectId : `int` or iterable of `int`
            Restrict to sources of this object, or of any of these objects
            (issued as chunked ``diaObjectId IN (...)`` queries). Required.
        mjd_begin, mjd_end : `float`, optional
            Restrict to ``mjd_begin <= midpointMjdTai < mjd_end`` (TAI).
        bands : `list` [`str`], optional
            Restrict to these bands (subset of ``ugrizy``).
        columns : `list` [`str`], optional
            Columns to select; defaults to all columns.
        limit : `int`, optional
            Maximum number of rows to return; None means no limit.
        id_chunk_size : `int`, optional
            Number of diaObjectIds per ``IN`` query when an iterable is given.

        Returns
        -------
        sources : `astropy.table.Table`
            The matching DiaSources, ordered by ``midpointMjdTai``.

        Raises
        ------
        ValueError
            If ``diaObjectId`` is None.
        """
        extra = self._source_filter_clauses(bands, mjd_begin, mjd_end)
        return self._load_source_like("DiaSource", diaObjectId, extra,
                                      columns, limit, id_chunk_size)

    def load_forced_sources(self, *, diaObjectId, mjd_begin=None,
                            mjd_end=None, bands=None, columns=None,
                            limit=DEFAULT_ROW_LIMIT, id_chunk_size=1000):
        """Load DiaForcedSources, with the same filters as `load_sources`.

        Like `load_sources`, ``diaObjectId`` is required and there is no
        spatial argument; use `load_forced_sources_for_region`, or the
        prototype-only `load_forced_sources_by_cone` /
        `load_forced_sources_by_time_window`.

        Returns
        -------
        forced_sources : `astropy.table.Table`
            The matching DiaForcedSources, ordered by ``midpointMjdTai``.

        Raises
        ------
        ValueError
            If ``diaObjectId`` is None.
        """
        extra = self._source_filter_clauses(bands, mjd_begin, mjd_end)
        return self._load_source_like("DiaForcedSource", diaObjectId, extra,
                                      columns, limit, id_chunk_size)

    def load_sources_for_region(self, *, ra=None, dec=None, radius=None,
                                exposure=None, padding=DEFAULT_PADDING_ARCSEC,
                                mjd_begin=None, mjd_end=None, bands=None,
                                columns=None, limit=DEFAULT_ROW_LIMIT,
                                object_limit=DEFAULT_ROW_LIMIT,
                                id_chunk_size=1000):
        """Load DiaSources for a region, the production-compatible way.

        Performs the two-step workflow the production PPDB requires: cone-
        search DiaObjects (the only table that allows spatial search), then
        load the DiaSources of those objects.

        Parameters
        ----------
        ra, dec, radius : `float`, optional
            Cone center and radius, in degrees; mutually exclusive with
            ``exposure``. The radius is applied to *object* positions.
        exposure : `lsst.afw.image.Exposure`, optional
            Load sources for objects covering this exposure's footprint.
        padding : `float`, optional
            Margin around the exposure footprint, in arcseconds.
        mjd_begin, mjd_end, bands, columns : optional
            Passed through to the source query (see `load_sources`).
        limit : `int`, optional
            Cap on returned sources; None means no limit.
        object_limit : `int`, optional
            Cap on the DiaObjects found by the cone search.
        id_chunk_size : `int`, optional
            diaObjectIds per ``IN`` query when fetching sources.

        Returns
        -------
        sources : `astropy.table.Table`
            DiaSources of the objects in the region, ordered by time.

        Notes
        -----
        Because this is object-first, DiaSources not associated with a
        returned DiaObject (e.g. unassociated or ssObject-only detections)
        are not included. To find those, use `load_sources_by_cone`.
        """
        ids = self._region_object_ids(
            ra, dec, radius, exposure, padding, object_limit)
        return self.load_sources(diaObjectId=ids, mjd_begin=mjd_begin,
                                 mjd_end=mjd_end, bands=bands, columns=columns,
                                 limit=limit, id_chunk_size=id_chunk_size)

    def load_forced_sources_for_region(self, *, ra=None, dec=None,
                                       radius=None, exposure=None,
                                       padding=DEFAULT_PADDING_ARCSEC,
                                       mjd_begin=None, mjd_end=None, bands=None,
                                       columns=None, limit=DEFAULT_ROW_LIMIT,
                                       object_limit=DEFAULT_ROW_LIMIT,
                                       id_chunk_size=1000):
        """Load DiaForcedSources for a region (object-first).

        Same two-step workflow and caveats as `load_sources_for_region`.

        Returns
        -------
        forced_sources : `astropy.table.Table`
            DiaForcedSources of the objects in the region, ordered by time.
        """
        ids = self._region_object_ids(
            ra, dec, radius, exposure, padding, object_limit)
        return self.load_forced_sources(
            diaObjectId=ids, mjd_begin=mjd_begin, mjd_end=mjd_end, bands=bands,
            columns=columns, limit=limit, id_chunk_size=id_chunk_size)

    def _region_object_ids(self, ra, dec, radius, exposure, padding,
                           object_limit):
        """Cone-search DiaObjects and return their diaObjectIds."""
        objects = self.load_objects(ra=ra, dec=dec, radius=radius,
                                    exposure=exposure, padding=padding,
                                    columns=["diaObjectId"],
                                    limit=object_limit)
        if not len(objects):
            return []
        return objects["diaObjectId"].tolist()

    def load_sources_by_cone(self, *, ra=None, dec=None, radius=None,
                             exposure=None, padding=DEFAULT_PADDING_ARCSEC,
                             mjd_begin=None, mjd_end=None, bands=None,
                             columns=None, limit=DEFAULT_ROW_LIMIT):
        """Directly cone-search the DiaSource table (prototype-only).

        .. warning::

           The production PPDB will not allow spatial searches on the source
           tables. This works only against the prototype and is intended for
           debugging/validation (e.g. finding DiaSources with no associated
           DiaObject). For the production-compatible region workflow use
           `load_sources_for_region`.

        Parameters mirror `load_sources_for_region` minus ``object_limit``.

        Returns
        -------
        sources : `astropy.table.Table`
            DiaSources within the cone, ordered by ``midpointMjdTai``.
        """
        return self._load_by_cone(
            "DiaSource", "load_sources_for_region", ra=ra, dec=dec,
            radius=radius, exposure=exposure, padding=padding, bands=bands,
            mjd_begin=mjd_begin, mjd_end=mjd_end, columns=columns, limit=limit)

    def load_forced_sources_by_cone(self, *, ra=None, dec=None, radius=None,
                                    exposure=None,
                                    padding=DEFAULT_PADDING_ARCSEC,
                                    mjd_begin=None, mjd_end=None, bands=None,
                                    columns=None, limit=DEFAULT_ROW_LIMIT):
        """Directly cone-search the DiaForcedSource table (prototype-only).

        .. warning::

           Not supported by the production PPDB; prototype/debugging only.
           Prefer `load_forced_sources_for_region`.

        Returns
        -------
        forced_sources : `astropy.table.Table`
            DiaForcedSources within the cone, ordered by ``midpointMjdTai``.
        """
        return self._load_by_cone(
            "DiaForcedSource", "load_forced_sources_for_region", ra=ra,
            dec=dec, radius=radius, exposure=exposure, padding=padding,
            bands=bands, mjd_begin=mjd_begin, mjd_end=mjd_end, columns=columns,
            limit=limit)

    def load_sources_by_time_window(self, *, mjd_begin=None, mjd_end=None,
                                    bands=None, columns=None,
                                    limit=DEFAULT_ROW_LIMIT):
        """Load DiaSources in a time window, any object (prototype-only).

        .. warning::

           The production PPDB will not allow object-unrestricted searches on
           the source tables. This works only against the prototype and is
           intended for debugging (e.g. inspecting everything processed in an
           MJD range). For the production-compatible workflow use
           `load_sources_for_region`.

        Parameters
        ----------
        mjd_begin, mjd_end : `float`, optional
            Time window on ``midpointMjdTai`` (TAI), as
            ``mjd_begin <= midpointMjdTai < mjd_end``. At least one bound is
            required (an unbounded source-table scan is not permitted).
        bands : `list` [`str`], optional
            Restrict to these bands (subset of ``ugrizy``).
        columns : `list` [`str`], optional
            Columns to select; defaults to all columns.
        limit : `int`, optional
            Maximum number of rows to return; None means no limit.

        Returns
        -------
        sources : `astropy.table.Table`
            DiaSources in the window, ordered by ``midpointMjdTai``.

        Raises
        ------
        ValueError
            If neither ``mjd_begin`` nor ``mjd_end`` is given.
        """
        return self._load_by_time_window(
            "DiaSource", "load_sources_for_region", mjd_begin=mjd_begin,
            mjd_end=mjd_end, bands=bands, columns=columns, limit=limit)

    def load_forced_sources_by_time_window(self, *, mjd_begin=None,
                                           mjd_end=None, bands=None,
                                           columns=None,
                                           limit=DEFAULT_ROW_LIMIT):
        """Load DiaForcedSources in a time window, any object (prototype-only).

        .. warning::

           Not supported by the production PPDB; prototype/debugging only.
           Prefer `load_forced_sources_for_region`. At least one of
           ``mjd_begin`` / ``mjd_end`` is required.

        Returns
        -------
        forced_sources : `astropy.table.Table`
            DiaForcedSources in the window, ordered by ``midpointMjdTai``.

        Raises
        ------
        ValueError
            If neither ``mjd_begin`` nor ``mjd_end`` is given.
        """
        return self._load_by_time_window(
            "DiaForcedSource", "load_forced_sources_for_region",
            mjd_begin=mjd_begin, mjd_end=mjd_end, bands=bands, columns=columns,
            limit=limit)

    def load_sources_for_object(self, diaObjectId, *, columns=None,
                                limit=DEFAULT_ROW_LIMIT):
        """Load all DiaSources associated with one diaObject.

        Returns
        -------
        sources : `astropy.table.Table`
            The object's DiaSources, ordered by ``midpointMjdTai``.
        """
        return self.load_sources(diaObjectId=diaObjectId, columns=columns,
                                 limit=limit)

    def load_forced_sources_for_object(self, diaObjectId, *, columns=None,
                                       limit=DEFAULT_ROW_LIMIT):
        """Load all DiaForcedSources associated with one diaObject.

        Returns
        -------
        forced_sources : `astropy.table.Table`
            The object's DiaForcedSources, ordered by ``midpointMjdTai``.
        """
        return self.load_forced_sources(diaObjectId=diaObjectId,
                                        columns=columns, limit=limit)

    def load_source(self, diaSourceId, *, columns=None):
        """Load a single DiaSource by id.

        Raises
        ------
        RuntimeError
            If the source does not exist.
        """
        return self._load_one("DiaSource", "diaSourceId", diaSourceId, columns)

    def load_forced_source(self, diaForcedSourceId, *, columns=None):
        """Load a single DiaForcedSource by id.

        Raises
        ------
        RuntimeError
            If the forced source does not exist.
        """
        return self._load_one("DiaForcedSource", "diaForcedSourceId",
                              diaForcedSourceId, columns)

    def _load_one(self, table_name, id_column, id_value, columns):
        """Load exactly one row from a table by id, raising if absent."""
        adql = (f"{self._select_clause(columns, None)} FROM ppdb.{table_name} "
                f"WHERE {id_column} = {int(id_value)}")
        table = self.run_query(adql)
        if len(table) == 0:
            raise RuntimeError(
                f"{id_column}={id_value} not found in ppdb.{table_name}")
        return table[0]

    # ------------------------------------------------------------------
    # Light-curve assembly
    # ------------------------------------------------------------------
    def load_light_curve(self, diaObjectId):
        """Assemble the full PPDB record for one diaObject.

        Loads the current DiaObject plus all of its DiaSources and
        DiaForcedSources, time-ordered.

        Parameters
        ----------
        diaObjectId : `int`
            Identifier of the object.

        Returns
        -------
        light_curve : `DiaObjectLightCurve`
            The object and its associated source time series.

        Raises
        ------
        RuntimeError
            If no current version of the object exists.
        """
        dia_object = self.load_object(diaObjectId)
        dia_sources = self.load_sources_for_object(diaObjectId)
        dia_forced_sources = self.load_forced_sources_for_object(diaObjectId)
        return DiaObjectLightCurve(
            diaObjectId=int(diaObjectId),
            dia_object=dia_object,
            dia_sources=dia_sources,
            dia_forced_sources=dia_forced_sources)

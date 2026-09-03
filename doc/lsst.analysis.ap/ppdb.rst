.. py:currentmodule:: lsst.analysis.ap

.. _lsst.analysis.ap-ppdb:

##################
Querying the PPDB
##################

``lsst.analysis.ap.ppdb`` loads PPDB catalogs — DiaObjects, DiaSources, and
DiaForcedSources — through the Rubin Science Platform TAP service, and
``lsst.analysis.ap.ppdbPlotUtils`` maps what it finds onto the sky.

The choice of TAP (ADQL over ``pyvo``) rather than a direct database connection is
deliberate: it is the interface scientists will have once the PPDB goes live, so
exercising these tools exercises that interface.
All loaders return `astropy.table.Table` objects, with columns in SDM-schema order —
the TAP service returns ``SELECT *`` columns alphabetically, so the loaders reorder
them into the schema's scientifically-grouped order unless you pass an explicit
``columns`` list.

.. _lsst.analysis.ap-ppdb-connecting:

Connecting
==========

The prototype PPDB lives at the ``data-int`` RSP environment,
``https://data-int.lsst.cloud/api/ppdbtap`` (``DEFAULT_PPDB_TAP_URL``).
You need a ``data-int`` RSP token with the ``read:tap`` scope; generate one at
https://data-int.lsst.cloud and put it in the ``RSP_TOKEN`` environment variable.
The same token works cross-environment, so tooling running on the USDF RSP reaches
the prototype with it.

.. code-block:: python

    from lsst.analysis.ap.ppdb import PpdbTap

    ppdb = PpdbTap()                    # reads RSP_TOKEN from the environment
    ppdb = PpdbTap(token=my_token)      # or pass it explicitly

The constructor raises `RuntimeError` with the token instructions if no token is
available.
Tokens are read only from the environment or the constructor and are never logged.
`PpdbTap.run_query` is available for arbitrary ADQL when the loaders do not cover
what you need.

.. _lsst.analysis.ap-ppdb-versioning:

Two things to know about the schema
===================================

**DiaObject is versioned.**
A single object accumulates multiple rows over time, and only the row with
``validityEndMjdTai IS NULL`` is the current version.
Every DiaObject query here applies that filter by default (``latest=True``) so you
cannot accidentally retrieve or double-count stale versions; ``latest=False`` returns
the full history and is rarely what you want.
DiaSource and DiaForcedSource are append-only and unversioned.

**The source tables are not spatially searchable in production.**
The production PPDB will permit cone searches only on ``DiaObject``.
Sources for a region must therefore be loaded object-first: cone-search the
DiaObjects, then load the sources of those objects.
`PpdbTap.load_sources_for_region` and `PpdbTap.load_forced_sources_for_region` do
exactly that and are the encouraged entry points.
`PpdbTap.load_sources` and `PpdbTap.load_forced_sources` require a ``diaObjectId``
and take no spatial argument at all.

Direct, object-unrestricted access to the source tables exists only through the
explicitly named prototype-only methods — `PpdbTap.load_sources_by_cone`,
`PpdbTap.load_forced_sources_by_cone`, `PpdbTap.load_sources_by_time_window`, and
`PpdbTap.load_forced_sources_by_time_window` — which warn when used.
They will not work against the production PPDB.
Their legitimate use is debugging that the object-first path cannot express: finding
DiaSources with no associated DiaObject, or inspecting everything processed in an MJD
range.

Every loader takes ``limit``, defaulting to ``DEFAULT_ROW_LIMIT`` (100 000 rows), as a
guard against accidentally pulling a multi-million-row table.
Pass ``limit=None`` to lift it.

.. _lsst.analysis.ap-ppdb-loading:

Loading catalogs
================

Cone searches take ``ra``, ``dec``, ``radius`` in degrees, or an exposure whose
footprint defines the region:

.. code-block:: python

    # Objects in a cone.
    objects = ppdb.load_objects(ra=53.1, dec=-28.1, radius=0.1)

    # Objects covering an exposure, with a 5 arcsecond margin.
    exposure = butler.get("preliminary_visit_image", visit=..., detector=...)
    objects = ppdb.load_objects(exposure=exposure, padding=5.0)

    # Sources in a region, the production-compatible way.
    sources = ppdb.load_sources_for_region(ra=53.1, dec=-28.1, radius=0.1,
                                           mjd_begin=60600.0, bands=["g", "r"])

`region_from_exposure` computes that region on its own if you want the numbers: it
returns the ``(ra, dec, radius)`` of the smallest circle centered on the exposure's
bounding-box center that contains all four corners, plus ``padding`` arcseconds.
A circumscribing circle rather than the exact polygon keeps the ADQL portable —
``CIRCLE`` is universally supported by TAP services, while ``POLYGON`` support and
winding conventions vary.

The source loaders share ``mjd_begin``/``mjd_end`` (a half-open window on
``midpointMjdTai``, TAI), ``bands`` (a subset of ``ugrizy``), ``columns``, and
``limit``, and always return rows ordered by ``midpointMjdTai``.
``diaObjectId`` accepts a scalar or an iterable; iterables are issued as chunked
``diaObjectId IN (...)`` queries, with ``id_chunk_size`` (default 1000) ids per query.

.. note::

   Because ``load_*_for_region`` is object-first, DiaSources not associated with a
   returned DiaObject — unassociated detections, or ssObject-only ones — are not
   included.
   Finding those is what `PpdbTap.load_sources_by_cone` is for.

Single rows come back as `astropy.table.Row`, and raise `RuntimeError` rather than
returning empty if the id does not exist: `PpdbTap.load_object` (current version
only), `PpdbTap.load_source`, `PpdbTap.load_forced_source`.

Light curves
------------

`PpdbTap.load_light_curve` assembles one object's complete PPDB record in a single
call and returns a `DiaObjectLightCurve`:

.. code-block:: python

    lc = ppdb.load_light_curve(diaObjectId)
    lc.dia_object          # astropy Row, the current DiaObject version
    lc.dia_sources         # Table, time-ordered
    lc.dia_forced_sources  # Table, time-ordered
    lc.n_sources, lc.n_forced_sources

`PpdbTap.load_sources_for_object` and `PpdbTap.load_forced_sources_for_object` are the
positional-argument shorthands for one object's sources when you do not need the whole
record.

.. _lsst.analysis.ap-ppdb-skydensity:

Mapping the PPDB on the sky
===========================

``ppdbPlotUtils.plot_ppdb_sky_density`` answers "what is actually in
the PPDB right now, and where?".
It bins a table into HEALPix pixels and draws the source density in # / deg² on a
``skyproj`` equal-area projection, with the galactic plane, an optional ecliptic, and
the LSST Deep Drilling Fields overlaid.
Because the projection is equal-area, the cos(dec) area factor is handled by the map
rather than by hand.

This module is not re-exported at package level; import it directly:

.. code-block:: python

    from lsst.analysis.ap.ppdbPlotUtils import plot_ppdb_sky_density

    sp, catalog = plot_ppdb_sky_density("DiaObject", ppdb=ppdb, nside=256)

Pass an existing `PpdbTap` as ``ppdb=`` to avoid re-authenticating.
The function returns the ``skyproj.Skyproj`` it drew into (``sp.ax`` is the underlying
axes) **and** the full unfiltered catalog it fetched — feed that back as ``catalog=``
to re-plot for free, which is the intended workflow: fetch the whole table once, then
zoom and retune thresholds without re-querying.

.. code-block:: python

    # Zoom in and tighten the cut, with no second query.
    sp, _ = plot_ppdb_sky_density("DiaObject", catalog=catalog,
                                  min_n_dia_sources=5,
                                  ra_lim=(-60, 60), dec_lim=(-40, 10))

The per-table filter is the one extra column worth fetching and cutting on:
``nDiaSources`` for DiaObject (``min_n_dia_sources``) and ``reliability`` for
DiaSource (``min_reliability``).
Both cuts are applied client-side, which is why they are free to change when
re-plotting.
Asking for the wrong table's filter raises rather than being silently ignored.
NaN reliabilities are dropped.

Other options: ``band_lines`` sets the galactic latitudes drawn as dashed lines
(default ``(10, -10)``, bracketing the ``|b| < 10`` plane band; pass ``()`` for none);
``draw_ecliptic`` and ``ecliptic_band_lines`` do the same for the ecliptic;
``label_fields`` marks and names the LSST fields inside the window;
``projection`` takes any equal-area ``skyproj.Skyproj`` subclass (the default is
``skyproj.McBrydeSkyproj``); and ``maxrec`` is the server-side row cap, which should sit
above the table size to avoid silent truncation.

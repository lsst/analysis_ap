.. py:currentmodule:: lsst.analysis.ap

.. _lsst.analysis.ap-overview:

########
Overview
########

``lsst.analysis.ap`` is a toolbox for *human-driven* inspection of alert-production
(AP) outputs: the difference images, the diaSource/diaObject catalogs written to the
APDB, and the PPDB that the APDB is eventually replicated into.
It is deliberately interactive — most entry points are functions and Tasks you call
from a notebook or a shell, not `~lsst.pipe.base.PipelineTask`\ s that run inside a
pipeline.
For metrics and plots generated automatically during a processing run, use
`analysis_tools <https://github.com/lsst/analysis_tools>`_ instead.

.. _lsst.analysis.ap-overview-map:

What lives where
================

The package splits into four groups of tools.

**Database access.**
:ref:`lsst.analysis.ap.apdb <lsst.analysis.ap-overview-databases>` (`DbQuery`,
`ApdbSqliteQuery`, `ApdbPostgresQuery`) reads an APDB directly over SQLAlchemy and
returns `pandas.DataFrame`\ s.
:doc:`lsst.analysis.ap.ppdb <ppdb>` (`PpdbTap`) reads the PPDB
over the Rubin Science Platform TAP service and returns `astropy.table.Table`\ s.
They are separate because the two databases are reached in fundamentally different
ways, and because the PPDB restricts which queries are legal.

**Image and cutout inspection.**
:doc:`plotImageSubtractionCutouts <cutouts>` renders per-diaSource
science/template/difference PNG cutouts (this is also the Zooniverse cutout
generator), and `PlotDiaSourceLightcurveTask` adds a lightcurve panel underneath
each one.
Both are available as command-line scripts.

**Notebook display and comparison helpers.**
:doc:`nb_utils <notebook-tools>` drives an interactive Firefly (or ds9) display:
`display_images` puts science/template/difference in three
locked frames with every AP catalog overlaid, and a family of ``compare_*`` /
``*_sharing_sources`` functions diffs two APDBs against each other to find
association changes.

**Run diagnostics.**
:doc:`taskRuntimes <runtimes>` summarizes per-task wall time and peak memory from the
``*_metadata`` datasets of a butler run.
`extract_timestamped_messages` flattens an LSST JSON log
into readable timestamped lines.

Several other modules (``compare``, ``imageQA``, ``plotUtils``,
``spatiallySampledMetricsQA``, ``apdbReconstruct``, ``skymapOverlay``) provide
supporting pieces; see the :ref:`Python API reference <lsst.analysis.ap-pyapi>`.

.. _lsst.analysis.ap-overview-databases:

APDB or PPDB?
=============

Almost every workflow in this package starts by connecting to one of the two alert
databases, so it is worth being clear on which one you want.

The **APDB** is the live, per-processing-run database.
It is what you have after running ``ap_pipe`` yourself, and it is usually either a
local SQLite file or a namespace in a shared Postgres instance:

.. code-block:: python

    from lsst.analysis.ap import ApdbSqliteQuery, ApdbPostgresQuery

    query = ApdbSqliteQuery("/path/to/association.db")
    # or, for a shared Postgres APDB:
    query = ApdbPostgresQuery("my_namespace",
                              "rubin@usdf-prompt-processing-dev.slac.stanford.edu/lsst-devl")

    objects = query.load_objects()             # pandas DataFrame
    sources = query.load_sources()
    lightcurve = query.load_sources_for_object(objects.diaObjectId.iloc[0])

`DbQuery.load_objects` returns only the current version of each diaObject by default
(``latest=True``), and `DbQuery.load_sources` takes an ``exclude_flagged`` argument.
The two SQL-backed subclasses additionally provide ``iter_sources``, which pages
through a large table and is what the cutout scripts use.

The **PPDB** is the long-lived, replicated database that scientists will query.
It is reached through TAP rather than a direct connection, on purpose: exercising
`~lsst.analysis.ap.ppdb.PpdbTap` exercises the same interface users will have.
See :doc:`ppdb` for the access token, the versioned ``DiaObject`` table, and the
object-first query pattern the production PPDB requires.

.. _lsst.analysis.ap-overview-guides:

Where to go next
================

- :doc:`notebook-tools` — display images with catalog overlays; compare two APDBs.
- :doc:`cutouts` — generate diaSource cutout PNGs, with or without lightcurves.
- :doc:`ppdb` — query the PPDB over TAP and map its contents on the sky.
- :doc:`runtimes` — profile a pipeline run's task runtimes and memory.

.. py:currentmodule:: lsst.analysis.ap

.. _lsst.analysis.ap-cutouts:

#################################
DiaSource cutouts and lightcurves
#################################

Two Tasks render per-diaSource PNG images from a butler repository and an APDB:

`PlotImageSubtractionCutoutsTask`
    A 3-panel template/science/difference cutout per diaSource, optionally annotated
    with catalog metadata and optionally accompanied by a `Zooniverse`_ upload
    manifest.
    See :doc:`its task page <tasks/lsst.analysis.ap.PlotImageSubtractionCutoutsTask>`
    for the annotation layout and example images.

`PlotDiaSourceLightcurveTask`
    A subclass that adds a lightcurve panel below the cutouts, showing the whole
    diaSource history of the associated diaObject.

.. _Zooniverse: https://www.zooniverse.org/

Both write ``{diaSourceId}.png`` under ``output_path/images/``, and both are available
as command-line scripts:
:doc:`plotImageSubtractionCutouts <scripts/plotImageSubtractionCutouts>` and
:doc:`plotDiaSourceLightcurve <scripts/plotDiaSourceLightcurve>`.

.. _lsst.analysis.ap-cutouts-cli:

From the command line
=====================

The scripts connect to an APDB, page through its diaSources, and write cutouts.
Exactly one of ``--sqlitefile`` or ``--namespace`` is required, to select a SQLite
file or a Postgres namespace:

.. code-block:: bash

    plotImageSubtractionCutouts --sqlitefile association.db \
        --collections u/me/my_ap_run -j 8 --limit 800 \
        /repo/main ./cutouts

``--limit`` is the number of sources to process, or — with ``--all`` — the page size
used to chunk the whole table.
``-j``/``--jobs`` sets the multiprocessing pool size; keep ``--limit`` at least 100×
``-j`` so each worker gets a useful batch.
``--reliabilityMin``/``--reliabilityMax`` restrict the selection to a reliability
range, which is how you pull "the ones the classifier was unsure about".
``-C``/``--configFile`` loads a `PlotImageSubtractionCutoutsConfig`.

The lightcurve script takes the same database, collection, and reliability arguments
but no ``-j``: it runs single-process, because it caches lightcurves per diaObject and
multiprocessing would defeat the cache.
Passing ``njobs`` to the Task's ``run`` or ``write_images`` logs a warning and is
ignored.

.. _lsst.analysis.ap-cutouts-python:

From Python
===========

Construct the Task with an ``output_path`` and call
`~PlotImageSubtractionCutoutsTask.run` with a `pandas.DataFrame` of diaSources and a
butler created with the collections to read images from.
The DataFrame must have at least ``ra``, ``dec``, ``diaSourceId``, ``detector``,
``visit``, and ``instrument`` columns:

.. code-block:: python

    import lsst.daf.butler as dafButler
    from lsst.analysis.ap import (ApdbSqliteQuery,
                                  PlotImageSubtractionCutoutsConfig,
                                  PlotImageSubtractionCutoutsTask)

    butler = dafButler.Butler("/repo/main", collections="u/me/my_ap_run")
    query = ApdbSqliteQuery("association.db")

    config = PlotImageSubtractionCutoutsConfig()
    config.sizes = [30, 60]          # two cutout scales, stacked
    config.add_metadata = True
    task = PlotImageSubtractionCutoutsTask(config=config, output_path="./cutouts")

    sources = query.load_sources(limit=200)
    ids = task.run(sources, butler, njobs=4)

`~PlotImageSubtractionCutoutsTask.run` returns the diaSourceIds it succeeded on;
sources whose images could not be loaded are logged and dropped rather than raising, so
one missing dataset does not abort a long run.
When running single-process (``njobs=0``, the default), diaSources whose PNG already
exists are skipped, which makes a re-run after a partial failure cheap; the
multiprocessing path re-renders them.

It splits into `~PlotImageSubtractionCutoutsTask.write_images` (the PNGs) and
`~PlotImageSubtractionCutoutsTask.write_manifest` (the Zooniverse CSV); call the
former alone if you only want images.
`~PlotImageSubtractionCutoutsTask.generate_image` produces a single cutout as an
in-memory `io.BytesIO`, which is what to use to display one cutout in a notebook
without touching disk.

Configuration
-------------

The fields you are most likely to set (see
:ref:`the full list <lsst.pipe.tasks.characterizeImage.PlotImageSubtractionCutoutsTask-configs>`):

``sizes``
    List of cutout widths in pixels; more than one produces a row of panels per size.
``use_footprint``
    Size each cutout from the diaSource's footprint bounding box instead of ``sizes``.
    This reads the footprint-bearing detection catalog, which must be sorted by id.
``add_metadata``
    Annotate the image with coordinates, fluxes, and flags, coloring flagged
    measurements red and matching `~lsst.afw.display.Display` mask-plane colors.
``science_image_type``, ``diff_image_type``
    Dataset-type overrides for older processings, whose science image was ``calexp``
    or ``initial_pvi``.
``url_root``
    Base URL the images will be served from.
    Setting it is what causes ``manifest.csv`` to be written; leaving it unset skips
    the manifest entirely.
``chunk_size``
    Files per subdirectory (must be a power of 10), so a large run does not put
    millions of PNGs in one directory.
``save_as_numpy``
    Also write the raw cutout arrays under ``output_path/numpy/``, for when you want
    to measure the pixels rather than look at them.

`CutoutPath` implements that chunking and is useful on its own for finding the images
a previous run wrote:

.. code-block:: python

    from lsst.analysis.ap import CutoutPath

    path = CutoutPath("./cutouts", chunk_size=10000)
    path(diaSourceId, f"{diaSourceId}.png")   # full path
    path.exists(diaSourceId, f"{diaSourceId}.png")

.. _lsst.analysis.ap-cutouts-lightcurves:

Adding a lightcurve panel
=========================

`PlotDiaSourceLightcurveTask` takes an extra ``apdb_query`` argument — the `DbQuery`
it uses to load the diaObject's history — and the input DataFrame needs two more
columns, ``diaObjectId`` and ``midpointMjdTai``:

.. code-block:: python

    from lsst.analysis.ap import (PlotDiaSourceLightcurveConfig,
                                  PlotDiaSourceLightcurveTask)

    config = PlotDiaSourceLightcurveConfig()
    config.lightcurve_height = 3.0
    task = PlotDiaSourceLightcurveTask(config=config,
                                       output_path="./lightcurves",
                                       apdb_query=query)
    task.run(query.load_sources(limit=50), butler)

The panel plots the diaObject's diaSources with one marker
(``lightcurve_marker_source``, default ``o``) and overlays diaForcedSource
measurements for visits that have a forced measurement but *no* diaSource with a
different marker (``lightcurve_marker_forced_only``, default ``v``) — so a gap in the
detections is visibly distinguishable from a gap in the coverage.
With ``highlight_current_source`` (the default) the diaSource being cut out is marked
with a vertical line and an open ring, tying the panel to the images above it.

``lightcurve_exclude_flagged`` defaults to False so the panel's row count matches a
direct APDB query; set it True to drop diaSources hitting the bad-flag list.
DiaForcedSources are always loaded unfiltered.

Unassociated diaSources have a NaN ``diaObjectId``; those render an empty placeholder
panel rather than failing, as does a failed lightcurve query.
`PlotDiaSourceLightcurveConfig` inherits every field of
`PlotImageSubtractionCutoutsConfig`, so ``sizes``, ``add_metadata`` and the rest apply
unchanged.

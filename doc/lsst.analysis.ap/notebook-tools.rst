.. py:currentmodule:: lsst.analysis.ap

.. _lsst.analysis.ap-notebook-tools:

##############################
Notebook tools (``nb_utils``)
##############################

``lsst.analysis.ap.nb_utils`` is the interactive half of the package: functions meant to
be called from a notebook cell with a butler and a couple of data ids in hand.
Everything in ``__all__`` is re-exported from the top level, so
``from lsst.analysis.ap import display_images`` works.

The module covers three jobs:

#. :ref:`Displaying images with AP catalog overlays
   <lsst.analysis.ap-notebook-tools-display>` (`display_images`,
   `display_images_ab`, `display_footprints`, `display_coadd_coverage`).
#. :ref:`Comparing two APDBs <lsst.analysis.ap-notebook-tools-compare>`
   (`compare_sources`, `compare_objects`, `classify_association_clusters`,
   `find_objects_sharing_sources`) and
   :ref:`plotting the disagreements <lsst.analysis.ap-notebook-tools-clusters>`
   (`plot_cutouts_with_object_markers`, `plot_objects_sharing_sources`).
#. :ref:`Small utilities <lsst.analysis.ap-notebook-tools-utilities>`
   (`make_simbad_link`, `get_xy_from_source_table`,
   `extract_timestamped_messages`).

.. _lsst.analysis.ap-notebook-tools-display:

Displaying images with catalog overlays
=======================================

`display_images` is the main diagnostic view of a single ``(visit, detector)``.
It loads the science, template, and difference images, puts them in three frames,
overlays every AP catalog it can find, and pixel-locks the frames together:

.. code-block:: python

    import lsst.daf.butler as dafButler
    from lsst.analysis.ap import display_images

    butler = dafButler.Butler("/repo/main", collections="u/me/my_ap_run")
    display_images(butler, visit=2024120600049, detector=4)

The default backend is ``"firefly"``, which needs a Firefly server and the
``display_firefly`` package (both are set up for you on the RSP); ``"ds9"`` also works
but cannot render footprints.
Catalogs that are missing from the butler are skipped silently, so the same call is
usable against partial or failed runs.

The overlays, in the order the pipeline creates them, are:

=============================  =========  ==============================
catalog                        symbol     color
=============================  =========  ==============================
psf-matching kernel sources    ``o``      green
unfiltered candidates          footprint  red (see ``unfiltered_style``)
rejected diaSources            ``+``      orange
long-trailed sources           ``x``      magenta
standardized diaSources        ``+``      blue
APDB, reliability > threshold  ``o``      blue, labeled with the score
APDB, reliability ≤ threshold  ``o``      red
solar-system matches           ``o``      cyan
marginal new diaSources        ``+``      yellow
=============================  =========  ==============================

Because the rows are in creation order, the last marker drawn at a pixel is the
pipeline's final word on that candidate; circle sizes step by two so nested ``o``
markers stay distinguishable.
A legend of what was actually found is printed to the notebook for each call.

Each overlay has its own ``show_*`` switch, so you can strip the display down to the
one question you are asking:

.. code-block:: python

    # Which candidates did filterDiaSource throw away, and why?
    display_images(butler, visit, detector,
                   show_apdb=False, show_standardized=False,
                   show_kernel_sources=False,
                   color_by=["pixelFlags_bad", "pixelFlags_edge",
                             "ip_diffim_DipoleFit_classification",
                             "pixelFlags_saturated"])

``color_by`` splits the ``dia_source_unfiltered`` overlay into buckets colored by the
first named flag that fires (list order is both color order and priority), with a
white residual bucket for rows matching none of them.

Two overlays are geometric rather than positional.
``show_dipoles`` draws a fixed-length white segment through each source classified as
a dipole, oriented along ``ip_diffim_DipoleFit_orientation``; the length is fixed
because the fitted separation is sub-pixel for nearly all of them, so the segment is a
marker of orientation, not of extent.
``show_trail_geometry`` draws a magenta segment along
``ext_trailedSources_Naive_angle`` at the measured trail length for trails longer than
3 px, and ``line_length_scale`` scales the drawn length (but not the 3 px threshold) so
short trails can be made visible.
Both come from the unfiltered catalog, because ``filterDiaSource`` removes dipoles and
long trails before the later stages.

Other options worth knowing:

``unfiltered_style``
    ``"footprint"`` (default) draws real `~lsst.afw.detection.Footprint` outlines via
    Firefly's native footprint rendering; ``"marker"`` draws the older ``+`` markers.
    Non-Firefly backends fall back to markers.
    Firefly gives no way to enumerate or delete footprint layers, so switching styles
    or re-running with ``color_by`` can leave stale layers on the frame — the function
    warns when that is a risk.
``use_fakes``
    Loads the fake-injected science (``fakes_``) and template
    (``injectedTemplate_``) images; the difference image and all catalogs keep their
    ordinary names, matching the fake-source pipeline's output convention.
``skymap``
    Pass a `~lsst.skymap.BaseSkyMap` to outline and label every tract/patch touching
    each frame.
``image_datasets``
    Override the ``{"science", "template", "difference"}`` → dataset-name mapping.
    The defaults (``preliminary_visit_image``, ``template_detector``,
    ``difference_image``) follow ``ApPipe.yaml``; older processings need
    e.g. ``science="calexp"``.
``dry_run``
    Load everything and print the legends, but do not open a display.
    The cheap way to see which datasets exist for a data id.

Comparing two runs
------------------

`display_images_ab` loads the same ``(visit, detector)`` from two butlers, shows one
image type in two frames, and overlays each butler's own catalogs on its own frame —
the A/B view for a config change that affects detection or subtraction:

.. code-block:: python

    from lsst.analysis.ap import display_images_ab

    display_images_ab(butler_before, butler_after, visit, detector,
                      image_type="difference", labels=("baseline", "new"))

Every keyword from `display_images` is accepted and applied to both frames.

Footprints on their own
-----------------------

`display_footprints` draws just the detection footprints, color-cycled by greedy graph
coloring over a bounding-box-touch graph so no two touching footprints share a color.
That makes it the tool for reading crowded blends, where outlines overlap and a single
color is unreadable:

.. code-block:: python

    from lsst.analysis.ap import display_footprints

    display_footprints(butler, visit=2024120600049, detector=4)
    # or draw a catalog you already have in hand:
    display_footprints(exposure=diffim, catalog=dia_source_unfiltered)

Footprints only exist on afw `~lsst.afw.table.SourceCatalog` datasets such as
``dia_source_unfiltered``; the transformed and APDB diaSource tables are DataFrames
with the footprints stripped, and passing one raises an explanatory `TypeError`.
This function is Firefly-only, and the color assignment is randomized among the
colors a footprint's neighbors are not using, so re-running gives a fresh coloring.

Template provenance
-------------------

`display_coadd_coverage` answers "which coadd patches actually went into this
template?".
Frame 0 is the difference image with each contributing patch outlined and labeled
``tract,patch``; frames 1..N are the patches themselves, each with the difference
image's footprint outlined on it.
The patch list comes from the template's ``coaddInputs`` records, so it is the set of
patches that supplied *valid pixels* — stricter than a skymap overlap lookup — and the
printed legend gives each patch's fractional overlap.

.. code-block:: python

    from lsst.analysis.ap import display_coadd_coverage

    # Just the provenance, no viewer:
    display_coadd_coverage(butler, visit, detector, dry_run=True)
    # Full patches, which is slow but shows the detector in context:
    display_coadd_coverage(butler, visit, detector, patch_extent="full")

Use ``patch_extent="overlap"`` with ``patch_margin`` to trim each coadd to the
difference image's footprint when the full-size patches are too slow to send to the
backend.
Frames are aligned by WCS (``align="Standard"``) here rather than by pixel, since the
coadds and the difference image do not share a pixel grid.

.. _lsst.analysis.ap-notebook-tools-compare:

Comparing two APDBs
===================

These functions diff two processing runs of the same data to find where association
changed.
They all take `DbQuery` handles or the catalogs those
handles return, never a connection string.

.. important::

   diaSourceIds are **not** stable between runs: they end in a per-catalog counter
   assigned in detection order, so any change to detection or measurement renumbers
   them.
   Every function here therefore pairs the two runs' diaSources *by position* (nearest
   neighbor within ``match_radius``, inside a single ``(visit, detector)``), and a
   detection with no counterpart in the other run cannot link objects across runs.

`compare_objects` and `compare_sources` are the coarse view — a spatial crossmatch
returning what is unique to each run and what matched:

.. code-block:: python

    from lsst.analysis.ap import ApdbSqliteQuery, compare_objects

    query1 = ApdbSqliteQuery("run1/association.db")
    query2 = ApdbSqliteQuery("run2/association.db")

    unique1, unique2, matched = compare_objects(query1, query2, match_radius=0.5)

``matched`` is the run-1 rows plus ``obj2_diaObjectId`` and ``xmatch_dist_arcsec``
columns (``src2_diaSourceId`` for `compare_sources`).
`compare_sources` can additionally generate and display cutouts of the sources unique
to each run, by way of `~lsst.analysis.ap.PlotImageSubtractionCutoutsTask`:

.. code-block:: python

    unique1, unique2, matched = compare_sources(
        butler1, butler2, query1, query2,
        make_cutouts=True, display_cutouts=True,
        cutout_path1="cutouts/run1", cutout_path2="cutouts/run2",
        njobs=4)

The two butlers must be created with the collections you want the cutout images read
from; they may be the same repository.

`classify_association_clusters` is the census.
It builds the bipartite graph of ``diaSource → run-1 diaObject`` and
``diaSource → run-2 diaObject`` edges over the diaSources the runs have in common,
extracts every connected component with union-find, and labels each one:

============  =============================================
``kind``      meaning
============  =============================================
``matched``   one run-1 object ↔ one run-2 object
``split``     one run-1 object became several in run 2
``merged``    several run-1 objects became one in run 2
``tangled``   M ↔ N, with both greater than one
============  =============================================

.. code-block:: python

    from lsst.analysis.ap import classify_association_clusters

    sources1, sources2 = query1.load_sources(), query2.load_sources()
    clusters = classify_association_clusters(sources1, sources2)
    print(clusters.kind.value_counts())            # all four kinds always present

    # Look at the worst offenders first.
    clusters[clusters.kind == "tangled"].sort_values("n_sources", ascending=False)

Each row carries ``n_obj1``/``n_obj2``, ``n_sources``, the ``obj1_ids``/``obj2_ids``
tuples, and the cluster's mean ``ra``/``dec``.
``kind`` is an ordered categorical, so ``value_counts()`` and ``groupby()`` report
zero-count kinds rather than dropping them.

`find_objects_sharing_sources` is the drill-down for a single diaObject: given a
run-1 diaObjectId (typically from ``unique1``), it grows the connected component of
diaSources and diaObjects reachable from it in both runs, catching arbitrarily deep
merge/split chains.

.. code-block:: python

    sources, related1, related2 = find_objects_sharing_sources(
        diaObjectId, sources1, sources2,
        query1.load_objects(), query2.load_objects(),
        max_distance_arcsec=2)

``max_distance_arcsec`` bounds the *search* to diaSources near the starting object;
all diaSources of the objects it finds come back regardless of that distance.

.. _lsst.analysis.ap-notebook-tools-clusters:

Plotting an association cluster
===============================

Having found a cluster, these two functions render it as matplotlib cutouts (inline,
independent of Firefly).

`plot_cutouts_with_object_markers` makes one cutout per diaSource, each marked with a
``x`` at the source the cutout is centered on, a ``+`` at every other cluster source
landing inside the frame, and one color-coded marker per distinct diaObjectId — the
same color meaning the same diaObject across the whole run:

.. code-block:: python

    import pandas as pd
    from lsst.analysis.ap import (find_objects_sharing_sources,
                                  plot_cutouts_with_object_markers)

    sources, ro1, ro2 = find_objects_sharing_sources(
        diaObjectId, sources1, sources2, objects1, objects2)
    plot_cutouts_with_object_markers(sources, butler1, pd.concat([ro1, ro2]),
                                     display_cutouts=True, size=51)

`plot_objects_sharing_sources` is the higher-level convenience: it calls
`find_objects_sharing_sources` for you and builds a single two-column figure, one row
per diaSource, left panel overlaid with run-1 diaObjects and right panel with run-2
diaObjects.
The same cutout image backs both panels of a row, so the only difference you see is
the association.

.. code-block:: python

    sources, ro1, ro2 = plot_objects_sharing_sources(
        diaObjectId, sources1, sources2, objects1, objects2, butler,
        column_labels=("baseline", "new"), output_path="cluster.png")

.. note::

   The two columns get *independent* palette assignments, so a shared color between
   the left and right panels does not mean a shared diaObject.
   Read each column against its own legend.

.. _lsst.analysis.ap-notebook-tools-utilities:

Utilities
=========

`make_simbad_link` displays a clickable Simbad cone-search link for a position and
returns the query results as an `astropy.table.Table` (or None, with a message, if
nothing is within the radius) — the quick check on whether a diaSource sits on a known
object:

.. code-block:: python

    from lsst.analysis.ap import make_simbad_link

    results = make_simbad_link(source.ra, source.dec, radius_arcsec=3.0)

`get_xy_from_source_table` converts a table's sky coordinates to pixel positions with
a WCS, returning an `astropy.table.Table` of ``x``/``y``.
It accepts either ``ra``/``dec`` (degrees) or ``coord_ra``/``coord_dec`` (radians) and
infers the units from whichever pair it finds, so it works on both DataFrame-style AP
tables and afw catalogs; pass ``degrees=`` to override.

`extract_timestamped_messages` flattens an LSST-style JSON log — the string or the
parsed dict — into ``asctime message`` lines, one per record, which is the readable
form for pasting into a ticket:

.. code-block:: python

    from lsst.analysis.ap import extract_timestamped_messages

    with open("pipetask-run.json") as f:
        print(extract_timestamped_messages(f.read()))

It is tolerant of the ways such logs arrive in practice: a bare JSON document, one
wrapped in an extra layer of quotes, or one that has been JSON-encoded twice.
Records missing ``asctime`` or ``message`` are skipped.

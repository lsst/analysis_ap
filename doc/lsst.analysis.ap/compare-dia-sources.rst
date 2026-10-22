.. py:currentmodule:: lsst.analysis.ap

.. _lsst.analysis.ap-compare-dia-sources:

########################################
Comparing the diaSources of two runs
########################################

`compare_dia_sources` matches the diaSources of two pipeline runs and explains
every source found in only one of them.
It reads the butler, so it works for runs that skipped association and have
no APDB; `compare_sources` needs an APDB for each run.

.. code-block:: python

    from lsst.daf.butler import Butler
    from lsst.analysis.ap import compare_dia_sources, plot_dia_source_comparison

    ref = Butler("/repo/main", collections="u/me/main_run")
    new = Butler("/repo/main", collections="u/me/ticket_run")

    result = compare_dia_sources(ref, new)
    result.summary          # band and source counts per visit
    result.reason_counts()  # why sources are missing from the other run
    plot_dia_source_comparison(result, labels=("main", "ticket"))

The two butlers' default collections select the runs; they can be the same
repository.
Pass ``visits=`` to compare only some visits.

Where the sources come from
===========================

Each run's diaSources are read from the visit table that
``consolidateDiaSourceTable`` writes (``dia_source_visit``; pass
``dataset="diaSourceTable"`` for DRP).
That task is in the ``afterburner`` subset of ``ApPipe.yaml``, so a run of the
``prompt`` subset alone has no visit table; for any visit without one, the
detector tables it would have consolidated (``dia_source_detector``) are read
instead.
``summary`` gives the band of each visit and records which table each run's
sources came from.

These tables hold only the sources that passed every filter, including the
reliability cut.
A source can be missing from one run because it was removed, not because it
was not detected; see :ref:`lsst.analysis.ap-compare-dia-sources-reasons`.

Matching
========

diaSource IDs differ between runs, so sources are matched by position.
Within each visit, every pair closer than ``radius`` (0.5 arcsec by default)
is a candidate, and pairs are taken in order of increasing separation,
skipping any whose sources are already paired; this is
`match_catalogs` with ``one_to_one=True``.
Matching across the whole visit lets a source that falls on a neighboring
detector in the other run still match.
The sign of a source is not used, so a source detected as positive in one run
and negative in the other still matches; ``flags_changed`` then lists
``isNegative``.

The results are tables:

``matched``
    One row per pair: both IDs and detectors, the separation and pixel offset,
    ``psfFlux``, ``psfFluxErr``, ``apFlux``, ``scienceFlux``, ``snr`` and
    ``reliability`` from each run (suffixes ``_ref`` and ``_new``),
    ``delta_psfFlux_sigma``, and ``flags_changed``, the boolean columns whose
    values differ.
``flag_changes``
    For each boolean column that changed in any pair, how many pairs have it
    set in one run only.
``ref_only``, ``new_only``
    The unmatched sources with all their columns, an ``instrument`` column,
    and the ``reason`` they are unmatched.
    These can be passed to `cutout_grid` or
    `PlotImageSubtractionCutoutsTask` to look at
    the sources.

.. _lsst.analysis.ap-compare-dia-sources-reasons:

Why a source is unmatched
=========================

For each unmatched source, the other run's catalogs are searched at its
position, on the same visit and detector.
The first reason in the table below that applies is recorded, and
``other_diaSourceId``, ``other_separation_arcsec`` and ``other_reliability``
describe the source it refers to.

=====================  =====================================================
``reason``             The other run
=====================  =====================================================
``matched_elsewhere``  has a source there, paired with a closer one (a
                       detection split in two, or two merged into one).
``low_reliability``    removed it below the reliability cut
                       (``dia_source_low_reliability``); ``other_reliability``
                       is its score.
``rejected``           removed it in ``filterDiaSource``
                       (``rejected_dia_source``).
``long_trail``         removed it as a long trail
                       (``long_trailed_source_detector``).
``removed_later``      detected it (``dia_source_unfiltered``) but it is in
                       none of the catalogs above.
``not_detected``       did not detect it.
``no_data``            wrote none of these catalogs for the detector and has
                       no sources on it.
``unknown``            did not write ``dia_source_unfiltered`` but wrote some
                       of the others.
=====================  =====================================================

``upstream_datasets`` and ``detection_dataset`` change the catalogs searched.

The catalogs are read only for detectors with unmatched sources, and only
until every source there has a reason.
The first catalog of removed detections (the reliability cut, by default) is
searched by position, since it explains most unmatched sources.
Any sources still unexplained are then found in ``dia_source_unfiltered``;
one that is not there was not detected, and one that is was removed by
whichever catalog lists its detection, which keeps its ID.
Pass ``explain_unmatched=False`` to skip the search.

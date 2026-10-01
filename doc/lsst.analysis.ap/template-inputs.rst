.. py:currentmodule:: lsst.analysis.ap

.. _lsst.analysis.ap-template-inputs:

###############
Template inputs
###############

`get_template_inputs` finds the images that make up the template at the
position of each diaSource, with their observation times and the fraction of
the template flux each one supplied.
`summarize_template_inputs` reduces that to an effective template epoch per
diaSource.

.. code-block:: python

    from lsst.daf.butler import Butler
    from lsst.analysis.ap import get_template_inputs, summarize_template_inputs

    butler = Butler("/repo/main", collections="u/me/my_ap_run", instrument="LSSTCam")
    sources = butler.get("dia_source_detector", visit=2025052000123, detector=42)

    inputs = get_template_inputs(butler, sources)
    summary = summarize_template_inputs(inputs)

The diaSource table needs ``diaSourceId``, ``visit``, ``detector``, ``ra``,
``dec`` and ``x``, ``y``; APDB, PPDB and ``dia_source_detector`` tables all
have them.

How the inputs are found
========================

The template (``template_detector`` by default) must be an
`lsst.images.DifferenceImage`.
Its ``templates`` component lists the coadds it was built from, each with a
butler dataset ID and its footprint on the science image.
Templates written as legacy Exposures have no such record and are rejected.

Each coadd is looked up by dataset ID, so its collection does not need to be
searched.
Its provenance component decides how its inputs are read:

=================  ========================================================
Component          Inputs at a position
=================  ========================================================
``coaddInputs``    Legacy coadds. Every input whose bounding box and valid
                   polygon contain the position, with its single coadd
                   weight. Times come from each input's ``VisitInfo``.
``provenance``     Cell coadds. The inputs of the cell containing the
                   position whose overlap polygon contains it, with the
                   weight recorded for that cell. Times come from the
                   ``visit`` dimension records.
=================  ========================================================

Where coadds overlap, the fractions follow `lsst.ip.diffim.GetTemplateTask`:
patches of a tract are averaged with per-pixel weights ``variance**-0.5``,
then tracts are averaged the same way.
This needs the variance of each overlapping coadd at the position, which is
read as a small cutout.

The output has one row per (diaSource, coadd, input), and ``weight`` sums to 1
for each diaSource.
An input that appears in two overlapping coadds has a row for each; sum over
``input_visit`` and ``input_detector`` for its total.

Limitations
===========

- The weights are evaluated at the diaSource position only.
  Warping the template and PSF matching mix in nearby template pixels, which
  can come from other coadds or inputs.
- Coadd provenance does not record per-pixel rejection, so an input whose
  pixel was masked, or clipped as an artifact by
  ``CompareWarpAssembleCoaddTask``, is still reported.

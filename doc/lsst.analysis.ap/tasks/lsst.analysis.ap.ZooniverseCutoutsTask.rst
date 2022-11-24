.. lsst-task-topic:: lsst.analysis.ap.ZooniverseCutoutsTask

#####################
ZooniverseCutoutsTask
#####################

``ZooniverseCutoutsTask`` is a task to generate 3-image cutouts (template, science, difference) of difference image sources loaded from a catalog or APDB.
The output images can be uploaded to a `Zooniverse`_ project, or used for investigating difference imaging performance.
This task reads the images from a butler repo, and writes its output as individual PNG files to the directory passed to the ``run`` method, or specified on the commandline.

``ZooniverseCutoutsTask`` is available on the command line as :doc:`zooniverseCutouts <../scripts/zooniverseCutouts>`.

.. _Zooniverse: https://www.zooniverse.org/

.. _lsst.analysis.ap.ZooniverseCutoutsTask-summary:

Processing summary
==================

``ZooniverseCutoutsTask`` runs this sequence of operations:

#. For each source in the input data, read the template, science, and difference image.

#. Cutout a region on each of those images (size determined by the config).

#. Make an image containing those cutout regions.

#. Write the image as a PNG to the specified output directory, using the ``diaSourceId`` as the filename.

.. _lsst.analysis.ap.ZooniverseCutoutsTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.analysis.ap.ZooniverseCutoutsTask

.. _lsst.analysis.ap.ZooniverseCutoutsTask-butler:

Butler datasets
===============

When run as the ``zooniverseCutouts`` command-line task, or directly through the `~lsst.analysis.ap.ZooniverseCutoutsTask.runDataRef` method, ``ZooniverseCutoutsTask`` obtains datasets from an input Butler data repository and a running APDB, but does not produce any output to the butler; the PNG image files are written to a specified directory.

.. _lsst.analysis.ap.ZooniverseCutoutsTask-butler-inputs:

Input datasets
--------------

``*_warpedExp``
    The template image that the template cutout is extracted from.

``calexp``
    The science image that the science cutout is extracted from.

``*_differenceExp``
    The difference image that the difference cutout is extracted from.

.. _lsst.analysis.ap.ZooniverseCutoutsTask-outputs:

Output datasets
===============

``ZooniverseCutoutsTask`` writes its output as PNG images to a specified directory, with the ``DiaSourceId`` of each source used as that PNG name.
These images are all of the same size and ordered as: template, science, difference.
The template and science iamges are min/max scaled with an ``asinh`` stretch, while the difference image is on a ``zscale`` with a linear stretch.

An example cutout image is shown below, in the format that would be most commonly used for a Zooniverse upload.

.. figure:: cutout_sample-plain.png
    :name: fig-cutout_sample-plain
    :alt: Basic image cutout, with no metadata.

    Basic image cutout, with no additional metadata.

The two below examples show what is produced when the ``addMetadata`` config field is set.
The first image shows a mocked image with no catalog flags set.
The source id, instrument, detector, visit, and filter name are given on the top row.
The ``PSF`` and ``ap`` flux fields are fluxes on the difference image, while the ``total`` flux field is the forced flux on the science image, and the AB magnitude computed from that forced flux.
Note that there is no colored text; this would represent a likely good difference image source measurement.

.. figure:: cutout_sample-noflags.png
    :name: fig-cutout_sample-noflags
    :alt: Cutout with metadata annotations, and no catalog flags set.

    Cutout with metadata annotations, and no catalog flags set.

The second image shows a mocked image with all catalog flags set, representing a source with image and/or measurement problems.
The catalog flags are colored to match the `~lsst.afw.display.Display` mask plane colors.
The ``INTERP``, ``SAT``, ``CR``, and ``SUS`` annotations are displayed if either that catalog flag, or the equivalent ``"*Center"`` flag is set (e.g. ``INTERP`` is displayed if either ``interpolated`` or ``interpolatedCenter`` is set).
Also note that the ``PSF``, ``ap`` and ``total`` text labels are all in red: this signifies that the measurement algorithms for the PSF fit, aperture flux measurement, and forced PSF flux all had a flag set.
This example shows all flags set to showcase their positions and colors in the image.
In general, only a subset of these flags will be shown for any given source, but they will always be in the same position and color.
Detailed flag descriptions will eventually be available in the `SDM Schema browser`_;
until then, look at ``data/association-flag-map.yaml`` in `lsst.ap.association` for more information on these flags.

.. figure:: cutout_sample-flags.png
    :name: fig-cutout_sample-flags
    :alt: Cutout with metadata annotations, and all catalog flags set.

    Cutout with metadata annotations, and all catalog flags set.

.. _SDM Schema browser: https://dm.lsst.org/sdm_schemas/browser/baseline.html#DiaSource


.. _lsst.pipe.tasks.characterizeImage.ZooniverseCutoutsTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.analysis.ap.ZooniverseCutoutsTask

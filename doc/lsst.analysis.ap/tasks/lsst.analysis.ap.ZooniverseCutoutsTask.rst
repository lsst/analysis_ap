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

.. _lsst.analysis.ap.ZooniverseCutoutsTask-subtasks:


Configuration fields
====================

.. lsst-task-config-fields:: lsst.analysis.ap.ZooniverseCutoutsTask

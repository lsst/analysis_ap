.. py:currentmodule:: lsst.analysis.ap

.. _lsst.analysis.ap-runtimes:

#####################
Task runtimes
#####################

`collect_task_runtimes` profiles a finished pipeline
run without any extra instrumentation: it walks every ``<task>_metadata`` dataset in a
butler collection, extracts the timing and memory fields with
`~lsst.pipe.base.resource_usage.QuantumResourceUsage.from_task_metadata`, and returns
a tidy per-task summary.

.. code-block:: python

    import lsst.daf.butler as dafButler
    from lsst.analysis.ap import collect_task_runtimes

    butler = dafButler.Butler("/repo/main")
    df = collect_task_runtimes(butler, "u/me/my_ap_run")

    # With a box plot of the per-quantum spread:
    df, fig = collect_task_runtimes(butler, "u/me/my_ap_run", plot=True)

One row per task comes back, sorted slowest-first by ``total_time_max``, with
``n_quanta``, the mean/min/max/std of ``total_time`` in seconds, and the same four
statistics for peak memory.
The memory columns are suffixed with their unit — ``memory_mean_GB`` or
``memory_mean_MB`` — chosen once for the whole table (GB if any task crosses 1 GB) so
the numbers stay comparable down the column.
An empty run returns an empty DataFrame rather than raising.

``threshold`` (default 1.0 s) filters out the noise, and it does so at task rather
than quantum granularity: a task is kept if **at least one** of its quanta reaches the
threshold, and then *all* of its quanta contribute to the summary.
That is what makes the mean/std meaningful — you see the task's real cross-quantum
variability instead of a statistic computed only over its slow outliers.

.. code-block:: python

    # Only the tasks with a quantum taking 10 s or more.
    df = collect_task_runtimes(butler, collections, threshold=10.0)

With ``plot=True`` the second return value is a `matplotlib.figure.Figure` holding a
horizontal box plot of per-quantum ``total_time``, slowest task at the top, with the
threshold marked.
The x-axis switches to log scale automatically when the spread exceeds 100× the
threshold, which it usually does in an AP run.
Pass ``ax=`` to draw into an existing axes instead of a new figure.

Quanta whose metadata is missing an expected timing field — an aborted or killed run —
are skipped rather than failing the call, so this works on a run that did not finish.

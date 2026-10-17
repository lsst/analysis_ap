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

"""Compare the diaSources of two pipeline runs without an APDB.

The sources come from the visit-level table that
``consolidateDiaSourceTable`` writes (``dia_source_visit`` in AP), or from the
detector-level tables it consolidates when a run has no visit table. Sources
are matched by position, since diaSource IDs differ between runs. A source
found in only one run is explained by looking for it in the other run's
earlier catalogs, which record the detections that were removed before the
table was written.
"""

__all__ = [
    "DEFAULT_UPSTREAM_DATASETS",
    "DiaSourceComparison",
    "compare_dia_sources",
    "plot_dia_source_comparison",
]

import dataclasses

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, match_coordinates_sky

from lsst.daf.butler import MissingDatasetTypeError

from .compare import match_catalogs

DEFAULT_UPSTREAM_DATASETS = {
    "low_reliability": "dia_source_low_reliability",
    "rejected": "rejected_dia_source",
    "long_trail": "long_trailed_source_detector",
}
"""Catalogs of removed detections searched for an unmatched source, keyed
by the reason recorded when the source is found in one (`dict` [`str`,
`str`]). The first, the most common reason, is searched by position; the
rest by the ID of the source's detection.
"""

_POSITION_SEARCHED = 1
"""Number of catalogs at the start of the upstream datasets searched by
position before the detection catalog is read (`int`). The rest are searched
by the ID of the source's detection.
"""

_COMPARED_COLUMNS = ("psfFlux", "psfFluxErr", "apFlux", "scienceFlux", "snr", "reliability")
"""Columns reported for both sources of a matched pair."""

_REASONS = ("matched_elsewhere", *DEFAULT_UPSTREAM_DATASETS, "removed_later",
            "not_detected", "no_data", "unknown")
"""Every reason the default datasets can give, in the order they are
tested.
"""

_GROUP_COLUMNS = ["instrument", "visit", "detector"]


@dataclasses.dataclass
class DiaSourceComparison:
    """The diaSources of two runs, matched by position.

    Attributes
    ----------
    summary : `pandas.DataFrame`
        One row per instrument and visit, with the band of the visit
        (``band``), the number of sources in each run (``n_ref``,
        ``n_new``), matched (``n_matched``) and in one run only
        (``n_ref_only``, ``n_new_only``), and the dataset each run's sources
        were read from (``table_ref``, ``table_new``; `None` if the run has
        no sources for the visit).
    matched : `pandas.DataFrame`
        One row per matched pair: the IDs and detectors of both sources,
        their separation (``separation_arcsec``), the pixel offset of the new
        source (``dx``, ``dy``; NaN if the detectors differ), each compared
        column with ``_ref`` and ``_new`` suffixes, the PSF flux difference in
        units of its uncertainty (``delta_psfFlux_sigma``), and the boolean
        columns whose values differ (``flags_changed``, comma separated;
        ``isNegative`` if the sign changed).
    ref_only, new_only : `pandas.DataFrame`
        Sources found in one run only, with all their columns plus
        ``reason``, and ``other_diaSourceId``, ``other_separation_arcsec``
        and ``other_reliability`` for the source the reason refers to (the
        ID is that catalog's, not a diaSourceId, for a removed detection).
    flag_changes : `pandas.DataFrame`
        One row per boolean column that differs in at least one pair, with
        the number of pairs where it is set in the reference run only
        (``n_ref_only``) and in the new run only (``n_new_only``).
    """

    summary: pd.DataFrame
    matched: pd.DataFrame
    ref_only: pd.DataFrame
    new_only: pd.DataFrame
    flag_changes: pd.DataFrame

    def reason_counts(self):
        """Count the unmatched sources of each run by reason.

        Returns
        -------
        counts : `pandas.DataFrame`
            Indexed by reason, with columns ``ref_only`` and ``new_only``.
        """
        counts = pd.DataFrame({
            "ref_only": self.ref_only["reason"].value_counts(),
            "new_only": self.new_only["reason"].value_counts(),
        }).fillna(0).astype(int)
        order = [reason for reason in _REASONS if reason in counts.index]
        return counts.loc[order + sorted(set(counts.index) - set(order))]


def _query_refs(butler, dataset, visits=None):
    """Return the refs of a dataset type, optionally for some visits only.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler to query.
    dataset : `str`
        Dataset type, with a ``visit`` dimension.
    visits : `dict` [`str`, `collections.abc.Iterable` [`int`]], optional
        Visits to find, keyed by instrument; all visits if not given.

    Returns
    -------
    refs : `list` [`lsst.daf.butler.DatasetRef`]
        The refs; none if the repository does not define the dataset type.
    """
    try:
        if visits is None:
            return list(butler.query_datasets(dataset, limit=None, explain=False))
        refs = []
        for instrument, instrument_visits in visits.items():
            instrument_visits = sorted({int(visit) for visit in instrument_visits})
            if instrument_visits:
                refs.extend(butler.query_datasets(
                    dataset, data_id={"instrument": instrument}, where="visit IN (:visits)",
                    bind={"visits": instrument_visits}, limit=None, explain=False))
        return refs
    except MissingDatasetTypeError:
        return []


def _read_table(butler, ref):
    """Read a diaSource table as a DataFrame with an ``instrument`` column."""
    table = butler.get(ref)
    frame = table.to_pandas() if hasattr(table, "to_pandas") else pd.DataFrame(table)
    frame["instrument"] = ref.dataId["instrument"]
    return frame


def _load_run(butler, dataset, detector_dataset, visits):
    """Read the diaSources of one run.

    Each visit is read from its ``dataset`` table, or from its
    ``detector_dataset`` tables if it has none.

    Returns
    -------
    sources : `pandas.DataFrame`
        The diaSources, with an ``instrument`` column.
    tables : `dict` [`tuple`, `str`]
        The dataset each ``(instrument, visit)`` was read from.
    bands : `dict` [`tuple`, `str`]
        The band of each ``(instrument, visit)``.
    """
    if visits is not None:
        # A visit ID alone does not name a visit, so look for it under every
        # instrument.
        instruments = [record.name for record in butler.query_dimension_records("instrument",
                                                                                explain=False)]
        visits = {instrument: list(visits) for instrument in instruments}
    frames = []
    tables = {}
    bands = {}
    for ref in _query_refs(butler, dataset, visits):
        frames.append(_read_table(butler, ref))
        tables[(ref.dataId["instrument"], ref.dataId["visit"])] = dataset
        bands[(ref.dataId["instrument"], ref.dataId["visit"])] = ref.dataId["band"]
    if visits is not None:
        visits = {instrument: [visit for visit in instrument_visits if (instrument, visit) not in tables]
                  for instrument, instrument_visits in visits.items()}
    if visits is None or any(visits.values()):
        for ref in _query_refs(butler, detector_dataset, visits):
            key = (ref.dataId["instrument"], ref.dataId["visit"])
            if tables.get(key, detector_dataset) == detector_dataset:
                frames.append(_read_table(butler, ref))
                tables[key] = detector_dataset
                bands[key] = ref.dataId["band"]
    if not frames:
        empty = pd.DataFrame(columns=["diaSourceId", "instrument", "visit", "detector", "ra", "dec"])
        return empty, tables, bands
    return pd.concat(frames, ignore_index=True), tables, bands


def _read_catalog(butler, ref):
    """Read the positions in a catalog of detections.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler to read from.
    ref : `lsst.daf.butler.DatasetRef` or `None`
        The catalog: an afw `~lsst.afw.table.SourceCatalog` or a table.

    Returns
    -------
    catalog : `pandas.DataFrame` or `None`
        Columns ``id``, ``ra`` and ``dec`` (degrees), and ``reliability``
        (NaN where the catalog has none); `None` if ``ref`` is `None`.
    """
    if ref is None:
        return None
    catalog = butler.get(ref)
    if hasattr(catalog, "schema"):
        # Take the few columns needed from the afw catalog; converting the
        # whole catalog to astropy costs more than reading it.
        if not catalog.isContiguous():
            catalog = catalog.copy(deep=True)
        names = set(catalog.schema.getNames())
    else:
        names = set(catalog.colnames)
    n = len(catalog)
    if "coord_ra" in names:
        ra = np.degrees(np.asarray(catalog["coord_ra"], dtype=float))
        dec = np.degrees(np.asarray(catalog["coord_dec"], dtype=float))
    else:
        ra = np.asarray(catalog["ra"], dtype=float)
        dec = np.asarray(catalog["dec"], dtype=float)
    return pd.DataFrame({
        "id": np.asarray(catalog["id"] if "id" in names else catalog["diaSourceId"]),
        "ra": ra,
        "dec": dec,
        "reliability": (np.asarray(catalog["reliability"], dtype=float) if "reliability" in names
                        else np.full(n, np.nan)),
    })


def _nearest(sources, catalog, radius):
    """Find the nearest catalog entry within ``radius`` of each source.

    Returns
    -------
    index : `numpy.ndarray`
        Row of ``catalog`` for each source, or -1 if none is within
        ``radius``.
    separation : `numpy.ndarray`
        Separation in arcseconds, NaN where there is no entry.
    """
    index = np.full(len(sources), -1)
    separation = np.full(len(sources), np.nan)
    if len(sources) == 0 or len(catalog) == 0:
        return index, separation
    idx, sep, _ = match_coordinates_sky(SkyCoord(sources["ra"].to_numpy(), sources["dec"].to_numpy(),
                                                 unit="deg"),
                                        SkyCoord(catalog["ra"].to_numpy(), catalog["dec"].to_numpy(),
                                                 unit="deg"))
    sep = sep.to(u.arcsec).value
    found = sep <= radius.to_value(u.arcsec)
    index[found] = idx[found]
    separation[found] = sep[found]
    return index, separation


def _explain_unmatched(unmatched, other, other_butler, radius, upstream_datasets,
                       detection_dataset):
    """Find why each unmatched source has no partner in the other run.

    Parameters
    ----------
    unmatched : `pandas.DataFrame`
        Sources of one run with no partner.
    other : `pandas.DataFrame`
        All diaSources of the other run.
    other_butler : `lsst.daf.butler.Butler`
        Butler for the other run.
    radius : `astropy.units.Quantity`
        Search radius.
    upstream_datasets : `dict` [`str`, `str`]
        Catalogs of removed detections, keyed by reason; see
        `compare_dia_sources`.
    detection_dataset : `str`
        Catalog of all detections.

    Returns
    -------
    unmatched : `pandas.DataFrame`
        Copy of ``unmatched`` with ``reason``, ``other_diaSourceId``,
        ``other_separation_arcsec`` and ``other_reliability`` columns.
    """
    result = unmatched.copy()
    result["reason"] = pd.Series(None, index=result.index, dtype=object)
    result["other_diaSourceId"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["other_separation_arcsec"] = np.nan
    result["other_reliability"] = np.nan

    def assign(rows, reason, catalog):
        """Record ``reason`` for the rows that have a catalog entry
        nearby, and return the rows that still have no reason.
        """
        index, separation = _nearest(rows, catalog, radius)
        hit = index >= 0
        labels = rows.index[hit]
        result.loc[labels, "reason"] = reason
        result.loc[labels, "other_diaSourceId"] = catalog["id"].to_numpy()[index[hit]]
        result.loc[labels, "other_separation_arcsec"] = separation[hit]
        result.loc[labels, "other_reliability"] = catalog["reliability"].to_numpy()[index[hit]]
        return result.loc[rows.index][result.loc[rows.index, "reason"].isna()]

    other_positions = pd.DataFrame({
        "id": other["diaSourceId"].to_numpy(),
        "ra": other["ra"].to_numpy(dtype=float),
        "dec": other["dec"].to_numpy(dtype=float),
        "reliability": (other["reliability"].to_numpy(dtype=float) if "reliability" in other
                        else np.full(len(other), np.nan)),
        "instrument": other["instrument"].to_numpy(),
        "visit": other["visit"].to_numpy(),
        "detector": other["detector"].to_numpy(),
    })
    # One query per catalog for every detector with an unmatched source.
    catalog_datasets = [*upstream_datasets.values(), detection_dataset]
    visits = {instrument: rows["visit"].unique()
              for instrument, rows in unmatched.groupby("instrument")}
    catalog_refs = {}
    for dataset in catalog_datasets:
        for ref in _query_refs(other_butler, dataset, visits):
            data_id = ref.dataId
            catalog_refs[dataset, data_id["instrument"], data_id["visit"], data_id["detector"]] = ref

    def read(dataset, key):
        """Read a catalog of the other run for an (instrument, visit,
        detector), or return `None` if it was not written.
        """
        return _read_catalog(other_butler, catalog_refs.get((dataset, *key)))

    upstream = list(upstream_datasets.items())
    for (instrument, visit, detector), rows in unmatched.groupby(_GROUP_COLUMNS):
        same_visit = other_positions[(other_positions["instrument"] == instrument)
                                     & (other_positions["visit"] == visit)]
        remaining = assign(rows, "matched_elsewhere", same_visit)
        if not len(remaining):
            continue
        key = (instrument, visit, detector)
        written = False
        for reason, dataset in upstream[:_POSITION_SEARCHED]:
            if (catalog := read(dataset, key)) is not None:
                written = True
                remaining = assign(remaining, reason, catalog)
        if not len(remaining):
            continue
        detections = read(detection_dataset, key)
        if detections is None:
            # Without the detection catalog, search the rest by position.
            for reason, dataset in upstream[_POSITION_SEARCHED:]:
                if len(remaining) and (catalog := read(dataset, key)) is not None:
                    written = True
                    remaining = assign(remaining, reason, catalog)
            if len(remaining):
                no_data = not written and not (same_visit["detector"] == detector).any()
                result.loc[remaining.index, "reason"] = "no_data" if no_data else "unknown"
            continue
        # A source absent from the detection catalog was not detected. One in
        # it was removed by whichever catalog lists its detection ID, since
        # removed detections keep their IDs.
        undetected = assign(remaining, "removed_later", detections)
        result.loc[undetected.index, "reason"] = "not_detected"
        unresolved = remaining.index.difference(undetected.index)
        for reason, dataset in upstream[_POSITION_SEARCHED:]:
            if unresolved.empty:
                break
            if (catalog := read(dataset, key)) is None:
                continue
            reliability = catalog.set_index("id")["reliability"]
            ids = result.loc[unresolved, "other_diaSourceId"]
            listed = ids.isin(reliability.index).to_numpy()
            labels = unresolved[listed]
            result.loc[labels, "reason"] = reason
            result.loc[labels, "other_reliability"] = reliability.loc[ids[listed].to_numpy()].to_numpy()
            unresolved = unresolved[~listed]
    return result


def _flag_columns(ref, new):
    """Return the boolean columns both tables have."""
    return [column for column in ref.columns
            if column in new.columns
            and pd.api.types.is_bool_dtype(ref[column]) and pd.api.types.is_bool_dtype(new[column])]


def _pair_table(pairs, ref, new):
    """Build the table of matched pairs and the flag-change counts."""
    partners = new.set_index("diaSourceId").loc[pairs["diaSourceId_2"].to_numpy()]
    matched = pd.DataFrame({
        "instrument": pairs["instrument"].to_numpy(),
        "visit": pairs["visit"].to_numpy(),
        "diaSourceId_ref": pairs["diaSourceId"].to_numpy(),
        "diaSourceId_new": partners.index.to_numpy(),
        "detector_ref": pairs["detector"].to_numpy(),
        "detector_new": partners["detector"].to_numpy(),
        "separation_arcsec": pairs["xmatch_dist_arcsec"].to_numpy(dtype=float),
    })
    same_detector = matched["detector_ref"] == matched["detector_new"]
    for axis in ("x", "y"):
        if axis in pairs and axis in partners:
            offset = partners[axis].to_numpy(dtype=float) - pairs[axis].to_numpy(dtype=float)
            matched[f"d{axis}"] = np.where(same_detector, offset, np.nan)
    for column in _COMPARED_COLUMNS:
        if column in pairs and column in partners:
            matched[f"{column}_ref"] = pairs[column].to_numpy(dtype=float)
            matched[f"{column}_new"] = partners[column].to_numpy(dtype=float)
    if {"psfFlux_ref", "psfFluxErr_ref", "psfFlux_new", "psfFluxErr_new"} <= set(matched.columns):
        sigma = np.hypot(matched["psfFluxErr_ref"], matched["psfFluxErr_new"])
        with np.errstate(divide="ignore", invalid="ignore"):
            matched["delta_psfFlux_sigma"] = np.where(
                sigma > 0, (matched["psfFlux_new"] - matched["psfFlux_ref"])/sigma, np.nan)

    flags = _flag_columns(ref, new)
    set_ref = pairs[flags].to_numpy(dtype=bool)
    set_new = partners[flags].to_numpy(dtype=bool)
    changed = set_ref != set_new
    matched["flags_changed"] = [",".join(np.asarray(flags)[row]) for row in changed]
    flag_changes = pd.DataFrame({
        "flag": flags,
        "n_ref_only": (set_ref & ~set_new).sum(axis=0),
        "n_new_only": (~set_ref & set_new).sum(axis=0),
    })
    flag_changes = flag_changes[(flag_changes["n_ref_only"] + flag_changes["n_new_only"]) > 0]
    flag_changes = flag_changes.sort_values(["n_ref_only", "n_new_only", "flag"],
                                            ascending=[False, False, True], ignore_index=True)
    return matched, flag_changes


def _summarize(ref, new, matched, ref_only, new_only, tables_ref, tables_new, bands):
    """Count the sources of each visit."""
    keys = sorted(set(tables_ref) | set(tables_new))

    def count(frame, columns=("instrument", "visit")):
        return frame.groupby(list(columns)).size() if len(frame) else pd.Series(dtype=int)

    counts = {
        "n_ref": count(ref),
        "n_new": count(new),
        "n_matched": count(matched),
        "n_ref_only": count(ref_only),
        "n_new_only": count(new_only),
    }
    rows = []
    for key in keys:
        row = {"instrument": key[0], "visit": key[1], "band": bands.get(key)}
        row.update({name: int(series.get(key, 0)) for name, series in counts.items()})
        row.update(table_ref=tables_ref.get(key), table_new=tables_new.get(key))
        rows.append(row)
    columns = ["instrument", "visit", "band", *counts, "table_ref", "table_new"]
    return pd.DataFrame(rows, columns=columns)


def compare_dia_sources(butler_ref, butler_new, *, dataset="dia_source_visit",
                        detector_dataset="dia_source_detector", visits=None, radius=0.5*u.arcsec,
                        explain_unmatched=True,
                        upstream_datasets=DEFAULT_UPSTREAM_DATASETS,
                        detection_dataset="dia_source_unfiltered"):
    """Match the diaSources of two runs and explain the differences.

    Parameters
    ----------
    butler_ref, butler_new : `lsst.daf.butler.Butler`
        Butlers whose default collections hold the reference and new runs.
    dataset : `str`, optional
        Visit-level diaSource table; ``diaSourceTable`` in DRP.
    detector_dataset : `str`, optional
        Detector-level diaSource table read for a visit that has no
        ``dataset`` table, as in a run that skipped
        ``consolidateDiaSourceTable``.
    visits : `collections.abc.Iterable` [`int`], optional
        Visits to compare; all visits if not given.
    radius : `astropy.units.Quantity`, optional
        Largest separation of a matched pair.
    explain_unmatched : `bool`, optional
        Look for each unmatched source in the other run's earlier catalogs
        to find why it has no partner. This reads those catalogs for every
        detector with an unmatched source. If False, every ``reason`` is
        `None`.
    upstream_datasets : `dict` [`str`, `str`], optional
        Catalogs of removed detections, keyed by the reason recorded when a
        source is found in one. The first is searched by position, so list
        the most common reason first; the rest are searched by the ID of the
        source's detection in ``detection_dataset``.
    detection_dataset : `str`, optional
        Catalog of all detections, whose IDs the removed detections keep. A
        source found only there has reason ``removed_later``; one absent from
        it has ``not_detected``.

    Returns
    -------
    comparison : `DiaSourceComparison`
        The matched pairs, unmatched sources and summaries.

    Raises
    ------
    LookupError
        Raised if neither run has any diaSources.

    Notes
    -----
    Sources are paired one to one within each visit, closest pairs first
    (`match_catalogs` with ``one_to_one=True``), so a source that lands on a
    neighboring detector in the other run still matches.

    The reason for an unmatched source is the first that applies:

    ``matched_elsewhere``
        The other run has a source within ``radius``, which was paired with
        a closer source (a split or merged detection).
    a key of ``upstream_datasets``
        The other run detected the source and removed it; by default
        ``low_reliability`` (``other_reliability`` is its score),
        ``rejected`` (by ``filterDiaSource``) or ``long_trail``.
    ``removed_later``
        The other run detected the source but it is in none of those
        catalogs.
    ``not_detected``
        The other run's detection catalog has no source there.
    ``no_data``
        The other run has none of these catalogs for the detector, nor any
        sources on it.
    ``unknown``
        The other run lacks the detection catalog but wrote others.
    """
    ref, tables_ref, bands_ref = _load_run(butler_ref, dataset, detector_dataset, visits)
    new, tables_new, bands_new = _load_run(butler_new, dataset, detector_dataset, visits)
    if len(ref) == 0 and len(new) == 0:
        raise LookupError(f"Neither run has any {dataset!r} or {detector_dataset!r} diaSources.")
    pairs, ref_only, new_only = match_catalogs(ref, new, radius=radius, on=("instrument", "visit"),
                                               id_col="diaSourceId", one_to_one=True)
    matched, flag_changes = _pair_table(pairs, ref, new)
    if explain_unmatched:
        ref_only = _explain_unmatched(ref_only, new, butler_new, radius, upstream_datasets,
                                      detection_dataset)
        new_only = _explain_unmatched(new_only, ref, butler_ref, radius, upstream_datasets,
                                      detection_dataset)
    else:
        ref_only = ref_only.assign(reason=None)
        new_only = new_only.assign(reason=None)
    # A visit's band is the same in both runs; the reference run's is used
    # where both have the visit.
    summary = _summarize(ref, new, matched, ref_only, new_only, tables_ref, tables_new,
                         bands_new | bands_ref)
    return DiaSourceComparison(summary=summary, matched=matched.reset_index(drop=True),
                               ref_only=ref_only.reset_index(drop=True),
                               new_only=new_only.reset_index(drop=True), flag_changes=flag_changes)


def plot_dia_source_comparison(comparison, labels=("ref", "new")):
    """Plot the differences between the diaSources of two runs.

    Parameters
    ----------
    comparison : `DiaSourceComparison`
        Result of `compare_dia_sources`.
    labels : `tuple` [`str`, `str`], optional
        Names of the reference and new runs.

    Returns
    -------
    figure : `matplotlib.figure.Figure`
        Four panels: PSF flux difference against signal to noise for
        matched pairs, reliability in each run for matched pairs, the
        number of unmatched sources by reason, and the sky positions of the
        unmatched sources.
    """
    from matplotlib.figure import Figure

    label_ref, label_new = labels
    matched = comparison.matched
    figure = Figure(figsize=(11, 9), layout="constrained")
    (flux_ax, reliability_ax), (reason_ax, sky_ax) = figure.subplots(2, 2)

    if "delta_psfFlux_sigma" in matched and "snr_ref" in matched:
        flux_ax.scatter(matched["snr_ref"], matched["delta_psfFlux_sigma"], s=6)
        flux_ax.set_xscale("log")
        for level in (-1, 0, 1):
            flux_ax.axhline(level, color="gray", lw=0.8, ls="-" if level == 0 else ":")
    flux_ax.set_xlabel(f"snr ({label_ref})")
    flux_ax.set_ylabel(f"psfFlux ({label_new} - {label_ref}) / sigma")
    flux_ax.set_title(f"{len(matched)} matched pairs")

    if "reliability_ref" in matched:
        reliability_ax.scatter(matched["reliability_ref"], matched["reliability_new"], s=6)
        reliability_ax.plot([0, 1], [0, 1], color="gray", lw=0.8)
    reliability_ax.set_xlabel(f"reliability ({label_ref})")
    reliability_ax.set_ylabel(f"reliability ({label_new})")

    counts = comparison.reason_counts()
    positions = np.arange(len(counts))
    reason_ax.barh(positions - 0.2, counts["ref_only"], height=0.4, label=f"{label_ref} only")
    reason_ax.barh(positions + 0.2, counts["new_only"], height=0.4, label=f"{label_new} only")
    reason_ax.set_yticks(positions, [str(reason) for reason in counts.index])
    reason_ax.invert_yaxis()
    reason_ax.set_xlabel("unmatched sources")
    reason_ax.legend()

    for frame, marker, label in ((comparison.ref_only, "o", label_ref),
                                 (comparison.new_only, "x", label_new)):
        sky_ax.scatter(frame["ra"], frame["dec"], s=12, marker=marker, label=f"{label} only")
    sky_ax.invert_xaxis()
    sky_ax.set_xlabel("ra (deg)")
    sky_ax.set_ylabel("dec (deg)")
    sky_ax.legend()
    return figure

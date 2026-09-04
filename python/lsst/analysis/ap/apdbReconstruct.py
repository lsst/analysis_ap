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

"""Reconstruct APDB-shaped catalogs from DiaPipelineTask butler outputs.

`lsst.ap.association.diaPipe.DiaPipelineTask` writes APDB-bound catalogs as
side-effects of its quantum execution: per-(visit, detector) DiaSource,
DiaObject, and DiaForcedSource datasets that mirror what gets inserted
into the APDB. `ApdbReconstructor` walks those datasets in a butler
collection and concatenates them into single DataFrames matching the APDB
SDM schema.

By default the reconstructor uses the dataset names configured by the
production AP pipeline (``ap_pipe/pipelines/_ingredients/ApPipe.yaml``'s
``associateApdb`` task: ``dia_source_apdb``, ``dia_object_apdb``,
``dia_forced_source_apdb``). For runs that use the raw
``DiaPipelineConnections`` defaults instead, pass ``dataset_names``
explicitly.
"""

from __future__ import annotations

__all__ = ["ApdbReconstruction", "ApdbReconstructor"]

import dataclasses
import logging

import pandas as pd

from lsst.pipe.tasks.schemaUtils import convertDataFrameToSdmSchema

from .apdb import DbQuery, _apdb_schema

_log = logging.getLogger(__name__)


@dataclasses.dataclass
class ApdbReconstruction:
    """APDB-shaped DataFrames reconstructed from DiaPipelineTask outputs.

    Attributes
    ----------
    diaSources : `pandas.DataFrame`
        DiaSource rows after association and standardization, deduped on
        ``diaSourceId``.
    diaObjects : `pandas.DataFrame`
        DiaObject rows. By default deduped to the latest snapshot per
        ``diaObjectId`` (``history=False`` in `ApdbReconstructor.reconstruct`).
    diaForcedSources : `pandas.DataFrame`
        DiaForcedSource rows, deduped on the schema primary key
        ``(diaObjectId, visit, detector)``.
    """
    diaSources: pd.DataFrame
    diaObjects: pd.DataFrame
    diaForcedSources: pd.DataFrame


class ApdbReconstructor:
    """Reconstruct APDB-shaped catalogs from `DiaPipelineTask` butler outputs.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler initialized with the collections that hold the pipeline run.
    dataset_names : `dict` [`str`, `list` [`str`]], optional
        Override the default dataset types used for each table. Keys are
        ``"diaSource"``, ``"diaObject"``, ``"diaForcedSource"``; each
        value is a list of dataset-type names that get concatenated and
        deduplicated. Defaults to ``DEFAULT_DATASET_NAMES``, which includes the
        ``preloaded_*`` datasets so the reconstruction includes both "history"
        rows from preload and the new ones.
    collections : `list` [`str`], optional
        Specify the butler collections to query if not using the default
        configured for the butler.
    where : `str`, optional
        Butler ``where`` clause passed through to ``queryDatasets``
        (e.g. ``"instrument='LSSTComCam' AND visit > 1000"``).
    """

    DEFAULT_DATASET_NAMES = {
        "diaSource": ["dia_source_apdb", "preloaded_dia_source"],
        "diaObject": ["dia_object_apdb", "preloaded_dia_object"],
        "diaForcedSource": ["dia_forced_source_apdb", "preloaded_dia_forced_source"],
    }

    def __init__(self, butler, dataset_names=None, *,
                 collections=None, where=None):
        self.butler = butler
        self.dataset_names = (dataset_names if dataset_names is not None
                              else self.DEFAULT_DATASET_NAMES)
        self.collections = collections
        self.where = where
        self.log = _log

    def _query_kwargs(self):
        kwargs = {"findFirst": True}
        if self.collections is not None:
            kwargs["collections"] = self.collections
        if self.where is not None:
            kwargs["where"] = self.where
        return kwargs

    def _load_tables(self, dataset_names):
        """Load and concatenate one or more dataset types into a single
        DataFrame. Returns an empty DataFrame if no dataset is present.

        Each dataset is loaded independently via `_load_table` and the
        results are concatenated. Dedup happens later in `finalize`, so
        overlap between sources (e.g. the same diaSource appearing in
        both ``dia_source_apdb`` and ``preloaded_dia_source`` after a
        prior pipeline run) is harmless.
        """
        frames = [self._load_table(name) for name in dataset_names]
        frames = [f for f in frames if len(f)]
        if not frames:
            return pd.DataFrame()
        if len(frames) == 1:
            return frames[0]
        return pd.concat(frames, ignore_index=True)

    def _load_table(self, dataset_name):
        """Load every instance of ``dataset_name`` from the butler and
        concatenate into a single DataFrame. Returns an empty DataFrame if
        the dataset type doesn't exist or no refs match.
        """
        try:
            refs = list(self.butler.registry.queryDatasets(
                dataset_name, **self._query_kwargs()))
        except Exception as e:
            self.log.info("Skipping %s: query failed (%s)",
                          dataset_name, e)
            return pd.DataFrame()
        if not refs:
            return pd.DataFrame()
        frames = [self.butler.get(ref, storageClass="DataFrame") for ref in refs]
        return pd.concat(frames, ignore_index=True)

    def reconstruct(self, *, coerce_to_schema=True, history=False):
        """Load all per-quantum catalogs and return APDB-shaped DataFrames.

        Parameters
        ----------
        coerce_to_schema : `bool`, optional
            If True (default), coerce each output to the SDM ``apdb.yaml``
            schema.
        history : `bool`, optional
            If True, keep every diaObject row written across all quanta.
            If False (default), dedupe to the atest row per ``diaObjectId``,
            mirroring the "current APDB state" view that ``DiaObjectLast``
            would give.

        Returns
        -------
        result : `ApdbReconstruction`
        """
        diaSources = self._load_tables(self.dataset_names["diaSource"])
        diaObjects = self._load_tables(self.dataset_names["diaObject"])
        diaForcedSources = self._load_tables(self.dataset_names["diaForcedSource"])
        return self.finalize(diaSources, diaObjects, diaForcedSources,
                             coerce_to_schema=coerce_to_schema,
                             history=history)

    @staticmethod
    def finalize(diaSources, diaObjects, diaForcedSources, *,
                 coerce_to_schema=True, history=False):
        """Dedup and (optionally) schema-coerce already-loaded catalogs.

        Parameters
        ----------
        diaSources, diaObjects, diaForcedSources : `pandas.DataFrame`
            Concatenated per-quantum catalogs.
        coerce_to_schema, history : see `reconstruct`.

        Returns
        -------
        result : `ApdbReconstruction`
        """
        # DiaSource: Primary Key (PK) is diaSourceId.
        if len(diaSources) and "diaSourceId" in diaSources.columns:
            diaSources = diaSources.drop_duplicates(subset="diaSourceId",
                                                    keep="last")
        # DiaForcedSource: PK is (diaObjectId, visit, detector).
        fkey = ["diaObjectId", "visit", "detector"]
        if (len(diaForcedSources)
                and set(fkey).issubset(diaForcedSources.columns)):
            diaForcedSources = diaForcedSources.drop_duplicates(
                subset=fkey, keep="last")
        # DiaObject: each quantum that touches a diaObject emits a row for
        # it. Many of those rows are "passthrough" snapshots: the diaObject
        # was in the quantum's preloaded working set but wasn't actually
        # updated, and the writer leaves ``nDiaSources`` (and other
        # update-only fields) as NaN/NULL. The validity timestamps on those
        # passthrough rows still advance to the quantum's processing time,
        # so a naive "sort by validityStart, keep last" dedup picks them
        # over the older snapshot that carries the real count.
        #
        # Fix: include ``nDiaSources`` as the primary sort key with NaN at
        # the front, so dedup ``keep="last"`` prefers any informative
        # snapshot (highest ``nDiaSources``, ties broken by latest validity)
        # over a passthrough one. Falls back to validity-only sort when
        # ``nDiaSources`` is absent.
        if (len(diaObjects) and "diaObjectId" in diaObjects.columns
                and not history):
            validity_col = next(
                (c for c in ("validityStartMjdTai", "validityStart")
                 if c in diaObjects.columns), None)
            sort_keys = []
            if "nDiaSources" in diaObjects.columns:
                sort_keys.append("nDiaSources")
            if validity_col is not None:
                sort_keys.append(validity_col)
            if sort_keys:
                diaObjects = diaObjects.sort_values(sort_keys,
                                                    na_position="first")
            diaObjects = diaObjects.drop_duplicates(subset="diaObjectId",
                                                    keep="last")

        if coerce_to_schema:
            schema = _apdb_schema()
            if len(diaSources):
                diaSources = convertDataFrameToSdmSchema(
                    schema, diaSources, "DiaSource", skipIndex=True)
            if len(diaObjects):
                diaObjects = convertDataFrameToSdmSchema(
                    schema, diaObjects, "DiaObject", skipIndex=True)
            if len(diaForcedSources):
                diaForcedSources = convertDataFrameToSdmSchema(
                    schema, diaForcedSources, "DiaForcedSource",
                    skipIndex=True)

        return ApdbReconstruction(
            diaSources=diaSources.reset_index(drop=True),
            diaObjects=diaObjects.reset_index(drop=True),
            diaForcedSources=diaForcedSources.reset_index(drop=True),
        )

    def to_query(self, *, coerce_to_schema=True, history=False):
        """Reconstruct and wrap the result as a `DbQuery`-compatible adapter
        so the in-memory frames can be passed to `lightcurve`,
        `PlotDiaSourceLightcurveTask`, and other tools that expect the
        ``DbQuery`` interface.

        Returns
        -------
        query : `InMemoryDbQuery`
        """
        recon = self.reconstruct(coerce_to_schema=coerce_to_schema,
                                 history=history)
        return InMemoryDbQuery(recon.diaSources,
                               recon.diaObjects,
                               recon.diaForcedSources)


class InMemoryDbQuery(DbQuery):
    """`DbQuery` backed by in-memory DataFrames (as from `ApdbReconstructor`).

    Implements the same load_* methods as `ApdbSqliteQuery`/`ApdbPostgresQuery`
    so the reconstructed data can be passed directly to ``lightcurve`` and
    ``PlotDiaSourceLightcurveTask``.
    """

    def __init__(self, diaSources, diaObjects, diaForcedSources):
        self._diaSources = diaSources
        self._diaObjects = diaObjects
        self._diaForcedSources = diaForcedSources
        self.diaSource_flags_exclude = []

    def set_excluded_diaSource_flags(self, flag_list):
        # Docstring inherited.
        missing = [f for f in flag_list if f not in self._diaSources.columns]
        if missing:
            raise ValueError(
                f"flag(s) {missing} not present in reconstructed DiaSource columns")
        self.diaSource_flags_exclude = list(flag_list)

    def _apply_flag_exclusion(self, df):
        if not self.diaSource_flags_exclude:
            return df
        mask = pd.Series(False, index=df.index)
        for flag in self.diaSource_flags_exclude:
            if flag in df.columns:
                mask |= df[flag].fillna(False).astype(bool)
        return df[~mask]

    def load_sources_for_object(self, dia_object_id, exclude_flagged=False,
                                limit=100000):
        # Docstring inherited.
        df = self._diaSources
        result = df[df["diaObjectId"] == dia_object_id]
        if exclude_flagged:
            result = self._apply_flag_exclusion(result)
        return result.head(limit).reset_index(drop=True)

    def load_forced_sources_for_object(self, dia_object_id,
                                       exclude_flagged=False, limit=100000):
        # Docstring inherited.
        df = self._diaForcedSources
        result = df[df["diaObjectId"] == dia_object_id]
        return result.head(limit).reset_index(drop=True)

    def load_source(self, id):
        # Docstring inherited.
        match = self._diaSources[self._diaSources["diaSourceId"] == id]
        if len(match) == 0:
            raise RuntimeError(f"diaSourceId={id} not found in DiaSource table")
        return match.iloc[0]

    def load_sources(self, exclude_flagged=False, limit=100000):
        # Docstring inherited.
        df = self._diaSources
        if exclude_flagged:
            df = self._apply_flag_exclusion(df)
        return df.head(limit).reset_index(drop=True)

    def load_object(self, id):
        # Docstring inherited.
        match = self._diaObjects[self._diaObjects["diaObjectId"] == id]
        if len(match) == 0:
            raise RuntimeError(f"diaObjectId={id} not found in DiaObject table")
        return match.iloc[0]

    def load_objects(self, limit=100000, latest=True):
        # Docstring inherited.
        return self._diaObjects.head(limit).reset_index(drop=True)

    def load_forced_source(self, id):
        # Docstring inherited..
        if "diaForcedSourceId" not in self._diaForcedSources.columns:
            raise RuntimeError("Reconstructed DiaForcedSource has no "
                               "diaForcedSourceId column")
        match = self._diaForcedSources[
            self._diaForcedSources["diaForcedSourceId"] == id]
        if len(match) == 0:
            raise RuntimeError(
                f"diaForcedSourceId={id} not found in DiaForcedSource table")
        return match.iloc[0]

    def load_forced_sources(self, limit=100000):
        # Docstring inherited.
        return self._diaForcedSources.head(limit).reset_index(drop=True)

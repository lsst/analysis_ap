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

from __future__ import annotations

__all__ = ["DbCassandraQuery"]

import os
import warnings
from typing import TYPE_CHECKING, cast

import pandas as pd

import lsst.utils
from lsst.ap.association import UnpackApdbFlags
from lsst.dax.apdb import Apdb, ApdbCassandra, ApdbTables
from lsst.dax.apdb.cassandra.cassandra_utils import quote_id
from lsst.pipe.base import Instrument
from lsst.resources import ResourcePathExpression
from .apdb import DbQuery

if TYPE_CHECKING:
    import lsst.daf.butler


class DbCassandraQuery(DbQuery):
    """Implementation of `DbQuery` interface for Cassandra backend.

    Parameters
    ----------
    config_uri : `~lsst.resources.ResourcePathExpression`
        URI or local file path pointing to a file with serialized
        configuration, or a string with a "label:" prefix to locate
        configuration in APDB index.
    butler : `lsst.daf.butler.Butler`
        Butler to unpack detector/visit from ccdVisitId.
        To be deprecated once this information is in the database.
    instrument : `str`
        Short name (e.g. "DECam") of instrument to make a dataId unpacker
        and to add to the table columns; supports any gen3 instrument.
        To be deprecated once this information is in the database.
    """

    timeout = 600
    """Timeout for queries in seconds. Regular timeout specified in APDB
    configuration could be too short for for full-scan queries that this
    class executes.
    """

    def __init__(
        self,
        config_uri: ResourcePathExpression,
        *,
        butler: lsst.daf.butler.Butler | None = None,
        instrument: Instrument | None = None,
    ):
        self._butler = butler
        self._instrument = instrument

        flag_map = os.path.join(
            lsst.utils.getPackageDir("ap_association"), "data/association-flag-map.yaml"
        )
        self._unpacker = UnpackApdbFlags(flag_map, "DiaSource")

        self.set_excluded_diaSource_flags(
            [
                "base_PixelFlags_flag_bad",
                "base_PixelFlags_flag_suspect",
                "base_PixelFlags_flag_saturatedCenter",
                "base_PixelFlags_flag_interpolated",
                "base_PixelFlags_flag_interpolatedCenter",
                "base_PixelFlags_flag_edge",
            ]
        )

        # We depend on ApdbCassandra for many things which we do not want to
        # reimplement here.
        apdb = Apdb.from_uri(config_uri)
        if not isinstance(apdb, ApdbCassandra):
            raise TypeError(
                f"Configuration file {config_uri} was produced for non-Cassandra backend."
            )
        self._apdb = apdb

    def set_excluded_diaSource_flags(self, flag_list: list[str]) -> None:
        # Docstring is inherited from base class.
        for flag in flag_list:
            if not self._unpacker.flagExists(flag, columnName="flags"):
                raise ValueError(f"flag {flag} not included in DiaSource flags")

        self.diaSource_flags_exclude = flag_list

    def _filter_flags(self, catalog: pd.DataFrame, column_name: str = "flags") -> None:
        """Filter catalog contents to exclude .

        Parameters
        ----------
        catalog : `pandas.DataFrame`
            Catalog to filter, update happens in-place.
        column_name : `str`, optional
            Name of flag column to query.
        """
        bitmask = int(
            self._unpacker.makeFlagBitMask(
                self.diaSource_flags_exclude, columnName=column_name
            )
        )
        if bitmask == 0:
            warnings.warn(
                f"Flag bitmask is zero. Supplied flags: {self.diaSource_flags_exclude}",
                RuntimeWarning,
            )
        mask = (catalog[column_name] & bitmask) != 0
        catalog.drop(catalog[mask].index, inplace=True)

    def _build_query(
        self,
        table: ApdbTables,
        *,
        columns: list[str] = [],
        where: str = "",
        limit: int = -1,
    ) -> str:
        """Build query for a specific table and selection."""
        if columns:
            what = ",".join(quote_id(column) for column in columns)
        else:
            what = "*"

        query = f'SELECT {what} from "{self._apdb._keyspace}"."{table.name}"'
        if where:
            query += f" WHERE {where}"
        if limit > 0:
            query += f" LIMIT {limit}"
        query += " ALLOW FILTERING"
        return query

    def load_sources_for_object(
        self, dia_object_id: int, exclude_flagged: bool = False, limit: int = 100000
    ) -> pd.DataFrame:
        # Docstring is inherited from base class.
        column_names = self._apdb._schema.apdbColumnNames(ApdbTables.DiaSource)
        query = self._build_query(
            ApdbTables.DiaSource,
            columns=column_names,
            where='"diaObjectId" = ?',
            limit=limit,
        )
        statement = self._apdb._preparer.prepare(query)
        result = self._apdb._session.execute(
            statement,
            (dia_object_id,),
            timeout=self.timeout,
            execution_profile="read_pandas",
        )
        catalog = cast(pd.DataFrame, result._current_rows)
        catalog.sort_values(by=["ccdVisitId", "diaSourceId"], inplace=True)

        if exclude_flagged:
            self._filter_flags(catalog)

        self._fill_from_ccdVisitId(catalog)

        return catalog

    def load_forced_sources_for_object(
        self, dia_object_id: int, exclude_flagged: bool = False, limit: int = 100000
    ) -> pd.DataFrame:
        # Docstring is inherited from base class.
        column_names = self._apdb._schema.apdbColumnNames(ApdbTables.DiaForcedSource)
        query = self._build_query(
            ApdbTables.DiaForcedSource,
            columns=column_names,
            where='"diaObjectId" = ?',
            limit=limit,
        )
        statement = self._apdb._preparer.prepare(query)
        result = self._apdb._session.execute(
            statement,
            (dia_object_id,),
            timeout=self.timeout,
            execution_profile="read_pandas",
        )
        catalog = cast(pd.DataFrame, result._current_rows)
        catalog.sort_values(by=["ccdVisitId", "diaForcedSourceId"], inplace=True)

        if exclude_flagged:
            self._filter_flags(catalog)

        self._fill_from_ccdVisitId(catalog)

        return catalog

    def load_source(self, id: int) -> pd.Series:
        # Docstring is inherited from base class.
        column_names = self._apdb._schema.apdbColumnNames(ApdbTables.DiaSource)
        query = self._build_query(
            ApdbTables.DiaSource, columns=column_names, where='"diaSourceId" = ?'
        )
        statement = self._apdb._preparer.prepare(query)
        result = self._apdb._session.execute(
            statement, (id,), timeout=self.timeout, execution_profile="read_pandas"
        )
        catalog = cast(pd.DataFrame, result._current_rows)
        self._fill_from_ccdVisitId(catalog)
        return catalog.iloc[0]

    def load_sources(
        self, exclude_flagged: bool = False, limit: int = 100000
    ) -> pd.DataFrame:
        # Docstring is inherited from base class.
        column_names = self._apdb._schema.apdbColumnNames(ApdbTables.DiaSource)
        query = self._build_query(
            ApdbTables.DiaSource,
            columns=column_names,
            limit=limit,
        )
        statement = self._apdb._preparer.prepare(query)
        result = self._apdb._session.execute(
            statement, timeout=self.timeout, execution_profile="read_pandas"
        )
        catalog = cast(pd.DataFrame, result._current_rows)
        catalog.sort_values(by=["ccdVisitId", "diaSourceId"], inplace=True)

        if exclude_flagged:
            self._filter_flags(catalog)

        self._fill_from_ccdVisitId(catalog)

        return catalog

    def load_object(self, id: int) -> pd.Series:
        # Docstring is inherited from base class.
        column_names = self._apdb._schema.apdbColumnNames(ApdbTables.DiaObjectLast)
        query = self._build_query(
            ApdbTables.DiaObjectLast, columns=column_names, where='"diaObjectId" = ?'
        )
        statement = self._apdb._preparer.prepare(query)
        result = self._apdb._session.execute(
            statement, (id,), timeout=self.timeout, execution_profile="read_pandas"
        )
        catalog = cast(pd.DataFrame, result._current_rows)
        return catalog.iloc[0]

    def load_objects(self, limit: int = 100000, latest: bool = True) -> pd.DataFrame:
        # Docstring is inherited from base class.
        if latest:
            table = ApdbTables.DiaObjectLast
        else:
            # when we do replication then we don't always generate DiaObject
            # contents.
            config = self._apdb.config
            if config.use_insert_id and config.use_insert_id_skips_diaobjects:
                raise ValueError("DiaObject history is not available for this database")
            table = ApdbTables.DiaObject

        column_names = self._apdb._schema.apdbColumnNames(table)
        query = self._build_query(
            table,
            columns=column_names,
            limit=limit,
        )
        statement = self._apdb._preparer.prepare(query)
        result = self._apdb._session.execute(
            statement, timeout=self.timeout, execution_profile="read_pandas"
        )
        catalog = cast(pd.DataFrame, result._current_rows)
        catalog.sort_values(by=["diaObjectId"], inplace=True)
        return catalog

    def load_forced_source(self, id: int) -> pd.Series:
        # Docstring is inherited from base class.
        column_names = self._apdb._schema.apdbColumnNames(ApdbTables.DiaForcedSource)
        query = self._build_query(
            ApdbTables.DiaForcedSource,
            columns=column_names,
            where='"diaForcedSourceId" = ?',
        )
        statement = self._apdb._preparer.prepare(query)
        result = self._apdb._session.execute(
            statement, (id,), timeout=self.timeout, execution_profile="read_pandas"
        )
        catalog = cast(pd.DataFrame, result._current_rows)
        self._fill_from_ccdVisitId(catalog)
        return catalog.iloc[0]

    def load_forced_sources(self, limit: int = 100000) -> pd.DataFrame:
        # Docstring is inherited from base class.
        column_names = self._apdb._schema.apdbColumnNames(ApdbTables.DiaForcedSource)
        query = self._build_query(
            ApdbTables.DiaForcedSource,
            columns=column_names,
            limit=limit,
        )
        statement = self._apdb._preparer.prepare(query)
        result = self._apdb._session.execute(
            statement, timeout=self.timeout, execution_profile="read_pandas"
        )
        catalog = cast(pd.DataFrame, result._current_rows)
        catalog.sort_values(by=["ccdVisitId", "diaForcedSourceId"], inplace=True)

        self._fill_from_ccdVisitId(catalog)

        return catalog

    def _fill_from_ccdVisitId(self, diaSources):
        """Expand the ccdVisitId value in the database.
        This method is temporary, until the CcdVisit table is filled out.

        Parameters
        ----------
        diaSources : `pandas.core.frame.DataFrame`
            Pandas dataframe with diaSources from an APDB; modified in-place.
        """
        if self._butler is None or self._instrument is None:
            return
        # do nothing for an empty series
        if len(diaSources) == 0:
            return
        instrumentDataId = self._butler.registry.expandDataId(
            instrument=self._instrument
        )
        packer = Instrument.make_default_dimension_packer(
            data_id=instrumentDataId, is_exposure=False
        )

        tempvisit = np.zeros(len(diaSources), dtype=np.int64)
        tempdetector = np.zeros(len(diaSources), dtype=np.int64)
        for i, ccdVisitId in enumerate(diaSources.ccdVisitId):
            dataId = packer.unpack(ccdVisitId)
            tempvisit[i] = dataId["visit"]
            tempdetector[i] = dataId["detector"]
        diaSources["visit"] = tempvisit
        diaSources["detector"] = tempdetector
        diaSources["instrument"] = self._instrument

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

"""APDB connection management and data access tools.
"""

__all__ = ["DbQuery", "ApdbSqliteQuery", "ApdbPostgresQuery"]

import os
import abc
import contextlib
import tempfile

import pandas as pd
import sqlalchemy
from sqlalchemy import or_, true
import numpy as np

from lsst.pipe.base import Instrument
from lsst.dax.apdb import Apdb, ApdbSql, ApdbTables


class DbQuery(abc.ABC):
    """Base class for APDB connection and query management.

    Subclasses must specify a ``connection`` property to use as a context-
    manager for queries.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler to unpack detector/visit from ccdVisitId.
        To be deprecated once this information is in the database.
    instrument : `str`
        Short name (e.g. "DECam") of instrument to make a dataId unpacker
        and to add to the table columns; supports any gen3 instrument.
        To be deprecated once this information is in the database.
    """

    def __init__(self, *, butler, instrument):
        self._butler = butler
        self._instrument = instrument
        self.set_excluded_diaSource_flags(['pixelFlags_bad',
                                           'pixelFlags_suspect',
                                           'pixelFlags_saturatedCenter',
                                           'pixelFlags_interpolated',
                                           'pixelFlags_interpolatedCenter',
                                           'pixelFlags_edge',
                                           ])

    @property
    @contextlib.contextmanager
    @abc.abstractmethod
    def connection(self):
        """Context manager for database connections.

        Yields
        ------
        connection : `sqlalchemy.engine.Connection`
            Connection to the database that will be queried. Whether the
            connection is closed after the context manager is closed is
            implementation dependent.
        """
        pass

    def set_excluded_diaSource_flags(self, flag_list):
        """Set flags of diaSources to exclude when loading diaSources.

        Any diaSources with configured flags are not returned
        when calling `load_sources_for_object` or `load_sources`
        with `exclude_flagged = True`.

        Parameters
        ----------
        flag_list : `list` [`str`]
            Flag names to exclude.
        """
        for flag in flag_list:
            if flag not in self._tables["DiaSource"].columns:
                raise ValueError(f"flag {flag} not included in DiaSource flags")

        self.diaSource_flags_exclude = flag_list

    def _make_flag_exclusion_query(self, query, table, flag_list):
        """Return an SQL where query that excludes sources with chosen flags.

        Parameters
        ----------
        flag_list : `list` [`str`]
            Flag names to exclude.
        query : `sqlalchemy.sql.Query`
            Query to include the where statement in.
        table : `sqlalchemy.schema.Table`
            Table containing the column to be queried.

        Returns
        -------
        query : `sqlalchemy.sql.Query`
            Query that selects rows to exclude based on flags.
        """
        # Build a query that selects any source with one or more chosen flags,
        # and return the opposite (`not_`) of that query.
        query = query.where(sqlalchemy.not_(sqlalchemy.or_(table.columns[flag_col] == 1
                                            for flag_col in flag_list)))
        return query

    def load_sources_for_object(self, dia_object_id, exclude_flagged=False, limit=100000):
        """Load diaSources for a single diaObject.

        Parameters
        ----------
        dia_object_id : `int`
            Id of object to load sources for.
        exclude_flagged : `bool`, optional
            Exclude sources that have selected flags set.
            Use `set_excluded_diaSource_flags` to configure which flags
            are excluded.
        limit : `int`
            Maximum number of rows to return.

        Returns
        -------
        data : `pandas.DataFrame`
            A data frame of diaSources for the specified diaObject.
        """
        table = self._tables["DiaSource"]
        query = table.select().where(table.columns["diaObjectId"] == dia_object_id)
        if exclude_flagged:
            query = self._make_flag_exclusion_query(query, table, self.diaSource_flags_exclude)
        query = query.order_by(table.columns["ccdVisitId"], table.columns["diaSourceId"])
        with self.connection as connection:
            result = pd.read_sql_query(query, connection)

        self._fill_from_ccdVisitId(result)
        return result

    def load_forced_sources_for_object(self, dia_object_id, exclude_flagged=False, limit=100000):
        """Load diaForcedSources for a single diaObject.

        Parameters
        ----------
        dia_object_id : `int`
            Id of object to load sources for.
        exclude_flagged : `bool`, optional
            Exclude sources that have selected flags set.
            Use `set_excluded_diaSource_flags` to configure which flags
            are excluded.
        limit : `int`
            Maximum number of rows to return.

        Returns
        -------
        data : `pandas.DataFrame`
            A data frame of diaSources for the specified diaObject.
        """
        table = self._tables["DiaForcedSource"]
        query = table.select().where(table.columns["diaObjectId"] == dia_object_id)
        if exclude_flagged:
            query = self._make_flag_exclusion_query(query, table, self.diaSource_flags_exclude)
        query = query.order_by(table.columns["ccdVisitId"], table.columns["diaForcedSourceId"])
        with self.connection as connection:
            result = pd.read_sql_query(query, connection)

        self._fill_from_ccdVisitId(result)
        return result

    def load_source(self, id):
        """Load one diaSource.

        Parameters
        ----------
        id : `int`
            The diaSourceId to load data for.

        Returns
        -------
        data : `pandas.Series`
            The requested diaSource.
        """
        table = self._tables["DiaSource"]
        query = table.select().where(table.columns["diaSourceId"] == id)
        with self.connection as connection:
            result = pd.read_sql_query(query, connection)
        if len(result) == 0:
            raise RuntimeError(f"diaSourceId={id} not found in DiaSource table")

        self._fill_from_ccdVisitId(result)
        return result.iloc[0]

    def load_sources(self, exclude_flagged=False, limit=100000):
        """Load diaSources.

        Parameters
        ----------
        exclude_flagged : `bool`, optional
            Exclude sources that have selected flags set.
            Use `set_excluded_diaSource_flags` to configure which flags
            are excluded.
        limit : `int`
            Maximum number of rows to return.

        Returns
        -------
        data : `pandas.DataFrame`
            All available diaSources.
        """
        table = self._tables["DiaSource"]
        query = table.select()
        if exclude_flagged:
            query = self._make_flag_exclusion_query(query, table, self.diaSource_flags_exclude)
        query = query.order_by(table.columns["ccdVisitId"], table.columns["diaSourceId"])
        if limit is not None:
            query = query.limit(limit)

        with self.connection as connection:
            result = pd.read_sql_query(query, connection)

        self._fill_from_ccdVisitId(result)
        return result

    def load_object(self, id):
        """Load the most-recently updated version of one diaObject.

        Parameters
        ----------
        id : `int`
            The diaObjectId to load data for.

        Returns
        -------
        data : `pandas.Series`
            The requested object.
        """
        table = self._tables["DiaObject"]
        query = table.select().where(table.columns["validityEnd"] == None)  # noqa: E711
        query = query.where(table.columns["diaObjectId"] == id)
        with self.connection as connection:
            result = pd.read_sql_query(query, connection)
        if len(result) == 0:
            raise RuntimeError(f"diaObjectId={id} not found in DiaObject table")

        return result.iloc[0]

    def load_objects(self, limit=100000, latest=True):
        """Load all diaObjects.

        Parameters
        ----------
        limit : `int`
            Maximum number of rows to return.
        latest : `bool`
            Only load diaObjects where validityEnd is None.
            These are the most-recently updated diaObjects.

        Returns
        -------
        data : `pandas.DataFrame`
            All available diaObjects.
        """
        table = self._tables["DiaObject"]
        if latest:
            query = table.select().where(table.columns["validityEnd"] == None)  # noqa: E711
        query = query.order_by(table.columns["diaObjectId"])
        if limit is not None:
            query = query.limit(limit)

        with self.connection as connection:
            result = pd.read_sql_query(query, connection)

        return result

    def load_forced_source(self, id):
        """Load one diaForcedSource.

        Parameters
        ----------
        id : `int`
            The diaForcedSourceId to load data for.

        Returns
        -------
        data : `pandas.Series`
            The requested forced source.
        """
        table = self._tables["DiaForcedSource"]
        query = table.select().where(table.columns["diaForcedSourceId"] == id)
        with self.connection as connection:
            result = pd.read_sql_query(query, connection)
        if len(result) == 0:
            raise RuntimeError(f"diaForcedSourceId={id} not found in DiaForcedSource table")

        self._fill_from_ccdVisitId(result)
        return result.iloc[0]

    def load_forced_sources(self, limit=100000):
        """Load all diaForcedSources.

        Parameters
        ----------
        limit : `int`
            Maximum number of rows to return.

        Returns
        -------
        data : `pandas.DataFrame`
            All available diaForcedSources.
        """
        table = self._tables["DiaForcedSource"]
        query = table.select()
        query = query.order_by(table.columns["ccdVisitId"], table.columns["diaForcedSourceId"])
        if limit is not None:
            query = query.limit(limit)

        with self.connection as connection:
            result = pd.read_sql_query(query, connection)
        self._fill_from_ccdVisitId(result)
        return result

    def _fill_from_ccdVisitId(self, diaSources):
        """Expand the ccdVisitId value in the database.
        This method is temporary, until the CcdVisit table is filled out.

        Parameters
        ----------
        diaSources : `pandas.core.frame.DataFrame`
            Pandas dataframe with diaSources from an APDB; modified in-place.
        """
        # do nothing for an empty series
        if len(diaSources) == 0:
            return
        instrumentDataId = self._butler.registry.expandDataId(instrument=self._instrument)
        packer = Instrument.make_default_dimension_packer(data_id=instrumentDataId,
                                                          is_exposure=False)

        tempvisit = np.zeros(len(diaSources), dtype=np.int64)
        tempdetector = np.zeros(len(diaSources), dtype=np.int64)
        for i, ccdVisitId in enumerate(diaSources.ccdVisitId):
            dataId = packer.unpack(ccdVisitId)
            tempvisit[i] = dataId['visit']
            tempdetector[i] = dataId['detector']
        diaSources['visit'] = tempvisit
        diaSources['detector'] = tempdetector
        diaSources['instrument'] = self._instrument


class ApdbSqliteQuery(DbQuery):
    """Open an sqlite3 APDB file to load data from it.

    This class keeps the sqlite connection open after initialization because
    our sqlite usage is to load a local file. Closing and re-opening would
    re-scan the whole file every time, and we don't need to worry about
    multiple users when working with local sqlite files.

    Parameters
    ----------
    filename : `str`
        Path to the sqlite3 file containing the APDB to load.
    butler : `lsst.daf.butler.Butler`
        Butler to unpack detector/visit from ccdVisitId.
        To be deprecated once this information is in the database.
    instrument : `str`
        Short name (e.g. "DECam") of instrument to make a dataId unpacker
        and to add to the table columns; supports any gen3 instrument.
        To be deprecated once this information is in the database.
    """

    def __init__(self, filename, *, butler, instrument, **kwargs):
        super().__init__(butler=butler, instrument=instrument, **kwargs)
        # For sqlite, use a larger pool and a faster timeout, to allow many
        # repeat transactions with the same connection, as transactions on
        # our sqlite DBs should be small and fast.
        self._engine = sqlalchemy.create_engine(f"sqlite:///{filename}",
                                                pool_timeout=5, pool_size=200)

        with self.connection as connection:
            metadata = sqlalchemy.MetaData()
            metadata.reflect(bind=connection)
        self._tables = metadata.tables

    @property
    @contextlib.contextmanager
    def connection(self):
        yield self._engine.connect()


class ApdbPostgresQuery(DbQuery):
    """Connect to a running postgres APDB instance and load data from it.

    This class connects to the database only when the ``connection`` context
    manager is entered, and closes the connection after it exits.

    Parameters
    ----------
    namespace : `str`
        Database namespace to load from. Called "schema" in postgres docs.
    url : `str`
        Complete url to connect to postgres database, without prepended
        ``postgresql://``.
    butler : `lsst.daf.butler.Butler`
        Butler to unpack detector/visit from ccdVisitId.
        To be deprecated once this information is in the database.
    instrument : `str`
        Short name (e.g. "DECam") of instrument to make a dataId unpacker
        and to add to the table columns; supports any gen3 instrument.
        To be deprecated once this information is in the database.
    """

    def __init__(self, namespace, url="rubin@usdf-prompt-processing-dev.slac.stanford.edu/lsst-devl",
                 *, butler, instrument, **kwargs):
        super().__init__(butler=butler, instrument=instrument, **kwargs)
        self._connection_string = f"postgresql://{url}"
        self._namespace = namespace
        self._engine = sqlalchemy.create_engine(self._connection_string, poolclass=sqlalchemy.pool.NullPool)

        with self.connection as connection:
            metadata = sqlalchemy.MetaData(schema=namespace)
            metadata.reflect(bind=connection)
        # ensure tables don't have schema prepended
        self._tables = {}
        for table in metadata.tables.values():
            self._tables[table.name] = table

    @property
    @contextlib.contextmanager
    def connection(self):
        _connection = self._engine.connect()
        try:
            yield _connection
        finally:
            _connection.close()

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

import abc
import contextlib
import warnings

import pandas as pd
import sqlalchemy


class DbQuery(abc.ABC):
    """Abstract interface for APDB queries.

    Notes
    -----
    APDB interface used by AP pipeline is defined by `lsst.dax.apdb.Apdb`
    class. Methods in this class are for non-pipeline tools that can analyse
    data produced by pipeline. APDB schema is not designed for analysis queries
    and performance of these methods can be non-optimal, especially for
    Cassandra backend. It is expected that these analysis queries should not be
    executed on production Cassandra service.
    """

    def set_excluded_diaSource_flags(self, flag_list: list[str]) -> None:
        """Set flags of diaSources to exclude when loading diaSources.

        Any diaSources with configured flags are not returned
        when calling `load_sources_for_object` or `load_sources`
        with `exclude_flagged = True`.

        Parameters
        ----------
        flag_list : `list` [`str`]
            Flag names to exclude.
        """
        raise NotImplementedError()

    def load_sources_for_object(
        self, dia_object_id: int, exclude_flagged: bool = False, limit: int = 100000
    ) -> pd.DataFrame:
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
        raise NotImplementedError()

    def load_forced_sources_for_object(
        self, dia_object_id: int, exclude_flagged: bool = False, limit: int = 100000
    ) -> pd.DataFrame:
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
        raise NotImplementedError()

    def load_source(self, id: int) -> pd.Series:
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
        raise NotImplementedError()

    def load_sources(self, exclude_flagged: bool = False, limit: int = 100000) -> pd.DataFrame:
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
        raise NotImplementedError()

    def load_object(self, id: int) -> pd.Series:
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
        raise NotImplementedError()

    def load_objects(self, limit: int = 100000, latest: bool = True) -> pd.DataFrame:
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
        raise NotImplementedError()

    def load_forced_source(self, id: int) -> pd.Series:
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
        raise NotImplementedError()

    def load_forced_sources(self, limit: int = 100000) -> pd.DataFrame:
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
        raise NotImplementedError()


class DbSqlQuery(DbQuery):
    """Base class for APDB connection and query management for SQL backends.

    Subclasses must specify a ``connection`` property to use as a context-
    manager for queries.

    Parameters
    ----------
    instrument : `str`
        Short name (e.g. "DECam") of instrument to make a dataId unpacker
        and to add to the table columns; supports any gen3 instrument.
        To be deprecated once this information is in the database.
    """

    def __init__(self, instrument=None):
        if instrument is not None:
            warnings.warn("The instrument name is now pulled from the APDB; "
                          "this kwarg is ignored and will be removed after v29",
                          FutureWarning,
                          stacklevel=2)

        self.set_excluded_diaSource_flags(['pixelFlags_bad',
                                           'pixelFlags_suspect',
                                           'pixelFlags_saturatedCenter',
                                           'pixelFlags_interpolated',
                                           'pixelFlags_interpolatedCenter',
                                           'pixelFlags_edge',
                                           ])

        key = "instrument"
        table = self._tables["metadata"]
        sql = sqlalchemy.sql.select(table.columns.value).where(table.columns.name == key)
        with self.connection as conn:
            result = conn.execute(sql)
            self._instrument = result.scalar()

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
        # Docstring is inherited.
        for flag in flag_list:
            if flag not in self._tables["DiaSource"].columns:
                raise ValueError(f"flag {flag} not included in DiaSource flags")

        self.diaSource_flags_exclude = flag_list

    def _make_flag_exclusion_query(self, query, table, flag_list):
        """Attach a where clause excluding sources with any chosen flag set.

        Parameters
        ----------
        query : `sqlalchemy.sql.Select`
            Query to attach the where clause to.
        table : `sqlalchemy.schema.Table`
            Reflected table containing the flag columns.
        flag_list : `list` [`str`]
            Flag column names to exclude.

        Returns
        -------
        query : `sqlalchemy.sql.Select`
            Query with the flag exclusion clause attached.
        """
        return query.where(sqlalchemy.and_(table.columns[col] == False  # noqa: E712
                                           for col in flag_list))

    def _load_table(self, table, *, where=None, exclude_flagged=False,
                    order_by=(), limit=None, fill_instrument=True):
        """Run a parameterized SELECT and return the result as a DataFrame.

        Parameters
        ----------
        table : `sqlalchemy.schema.Table`
            Reflected table to query.
        where : `sqlalchemy.sql.ClauseElement`, optional
            Extra where clause to attach.
        exclude_flagged : `bool`, optional
            If True, attach the configured flag-exclusion clause.
        order_by : `tuple` [`str`], optional
            Column names to order by.
        limit : `int`, optional
            Maximum number of rows to return; None means no limit.
        fill_instrument : `bool`, optional
            If True, append an ``instrument`` column to the result.

        Returns
        -------
        result : `pandas.DataFrame`
        """
        query = table.select()
        if where is not None:
            query = query.where(where)
        if exclude_flagged:
            query = self._make_flag_exclusion_query(query, table, self.diaSource_flags_exclude)
        if order_by:
            query = query.order_by(*[table.columns[c] for c in order_by])
        if limit is not None:
            query = query.limit(limit)
        with self.connection as connection:
            result = pd.read_sql_query(query, connection)
        if fill_instrument:
            self._fill_from_instrument(result)
        return result

    def _load_one(self, table_name, id_column, id_value, fill_instrument=True):
        """Load a single row from a table by id, raising if missing.

        Parameters
        ----------
        table_name : `str`
            Key into ``self._tables`` for the table to query.
        id_column : `str`
            Name of the id column to filter on.
        id_value : `int`
            Id value to match.
        fill_instrument : `bool`, optional
            If True, append an ``instrument`` column to the result.

        Returns
        -------
        row : `pandas.Series`

        Raises
        ------
        RuntimeError
            If no row matches.
        """
        table = self._tables[table_name]
        result = self._load_table(table,
                                  where=table.columns[id_column] == id_value,
                                  fill_instrument=fill_instrument)
        if len(result) == 0:
            raise RuntimeError(f"{id_column}={id_value} not found in {table_name} table")
        return result.iloc[0]

    def load_sources_for_object(self, dia_object_id, exclude_flagged=False, limit=100000):
        # Docstring is inherited.
        table = self._tables["DiaSource"]
        return self._load_table(
            table,
            where=table.columns["diaObjectId"] == dia_object_id,
            exclude_flagged=exclude_flagged,
            order_by=("visit", "detector", "diaSourceId"),
            limit=limit,
        )

    def load_forced_sources_for_object(self, dia_object_id, exclude_flagged=False, limit=100000):
        # Docstring is inherited.
        table = self._tables["DiaForcedSource"]
        return self._load_table(
            table,
            where=table.columns["diaObjectId"] == dia_object_id,
            exclude_flagged=exclude_flagged,
            order_by=("visit", "detector", "diaForcedSourceId"),
            limit=limit,
        )

    def load_source(self, id):
        # Docstring is inherited.
        return self._load_one("DiaSource", "diaSourceId", id)

    def load_sources(self, exclude_flagged=False, limit=100000):
        # Docstring is inherited.
        return self._load_table(
            self._tables["DiaSource"],
            exclude_flagged=exclude_flagged,
            order_by=("visit", "detector", "diaSourceId"),
            limit=limit,
        )

    def load_object(self, id):
        # Docstring is inherited.
        table = self._tables["DiaObject"]
        result = self._load_table(
            table,
            where=sqlalchemy.and_(
                table.columns["validityEnd"] == None,  # noqa: E711
                table.columns["diaObjectId"] == id,
            ),
            fill_instrument=False,
        )
        if len(result) == 0:
            raise RuntimeError(f"diaObjectId={id} not found in DiaObject table")
        return result.iloc[0]

    def load_objects(self, limit=100000, latest=True):
        # Docstring is inherited.
        table = self._tables["DiaObject"]
        where = table.columns["validityEnd"] == None if latest else None  # noqa: E711
        return self._load_table(
            table,
            where=where,
            order_by=("diaObjectId",),
            limit=limit,
            fill_instrument=False,
        )

    def load_forced_source(self, id):
        # Docstring is inherited.
        return self._load_one("DiaForcedSource", "diaForcedSourceId", id)

    def load_forced_sources(self, limit=100000):
        # Docstring is inherited.
        return self._load_table(
            self._tables["DiaForcedSource"],
            order_by=("visit", "detector", "diaForcedSourceId"),
            limit=limit,
        )

    def iter_sources(self, page_size=100000, reliability_min=None, reliability_max=None):
        """Yield DiaSources in pages of ``page_size`` rows.

        Parameters
        ----------
        page_size : `int`
            Number of rows per page.
        reliability_min, reliability_max : `float`, optional
            Inclusive bounds on the reliability column.

        Yields
        ------
        page : `pandas.DataFrame`
            One page of DiaSources, with the ``instrument`` column attached.
        """
        table = self._tables["DiaSource"]
        clauses = []
        if reliability_min is not None:
            clauses.append(table.columns["reliability"] >= reliability_min)
        if reliability_max is not None:
            clauses.append(table.columns["reliability"] <= reliability_max)
        where = sqlalchemy.and_(*clauses) if clauses else None

        offset = 0
        while True:
            query = table.select()
            if where is not None:
                query = query.where(where)
            query = query.order_by(table.columns["visit"],
                                   table.columns["detector"],
                                   table.columns["diaSourceId"])
            query = query.limit(page_size).offset(offset)
            with self.connection as connection:
                page = pd.read_sql_query(query, connection)
            if len(page) == 0:
                break
            self._fill_from_instrument(page)
            yield page
            offset += page_size

    def count_sources(self):
        """Return the total number of DiaSources in the database.

        Returns
        -------
        count : `int`
        """
        table = self._tables["DiaSource"]
        query = sqlalchemy.select(sqlalchemy.func.count()).select_from(table)
        with self.connection as connection:
            return connection.execute(query).scalar()

    def _fill_from_instrument(self, diaSources):
        """Add an instrument column to a list of sources.
        This method is temporary, until APDB has instrument in its metadata.

        Parameters
        ----------
        diaSources : `pandas.core.frame.DataFrame`
            Pandas dataframe with diaSources from an APDB; modified in-place.
        """
        # do nothing for an empty series
        if len(diaSources) == 0:
            return

        diaSources['instrument'] = self._instrument


class ApdbSqliteQuery(DbSqlQuery):
    """Open an sqlite3 APDB file to load data from it.

    This class keeps the sqlite connection open after initialization because
    our sqlite usage is to load a local file. Closing and re-opening would
    re-scan the whole file every time, and we don't need to worry about
    multiple users when working with local sqlite files.

    Parameters
    ----------
    filename : `str`
        Path to the sqlite3 file containing the APDB to load.
    instrument : `str`
        Short name (e.g. "DECam") of instrument to make a dataId unpacker
        and to add to the table columns; supports any gen3 instrument.
        To be deprecated once this information is in the database.
    """

    def __init__(self, filename, instrument=None, **kwargs):
        # For sqlite, use a larger pool and a faster timeout, to allow many
        # repeat transactions with the same connection, as transactions on
        # our sqlite DBs should be small and fast.
        self._engine = sqlalchemy.create_engine(f"sqlite:///{filename}",
                                                pool_timeout=5, pool_size=200)

        with self.connection as connection:
            metadata = sqlalchemy.MetaData()
            metadata.reflect(bind=connection)
        self._tables = metadata.tables
        super().__init__(**kwargs)

    @property
    @contextlib.contextmanager
    def connection(self):
        yield self._engine.connect()


class ApdbPostgresQuery(DbSqlQuery):
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
    instrument : `str`
        Short name (e.g. "DECam") of instrument to make a dataId unpacker
        and to add to the table columns; supports any gen3 instrument.
        To be deprecated once this information is in the database.
    """

    def __init__(self, namespace, url="rubin@usdf-prompt-processing-dev.slac.stanford.edu/lsst-devl",
                 instrument=None, **kwargs):
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
        super().__init__(instrument=instrument, **kwargs)

    @property
    @contextlib.contextmanager
    def connection(self):
        _connection = self._engine.connect()
        try:
            yield _connection
        finally:
            _connection.close()

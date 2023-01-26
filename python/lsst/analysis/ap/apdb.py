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
import warnings

import pandas as pd
import sqlalchemy

import lsst.utils
from lsst.ap.association import UnpackApdbFlags


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

        flag_map = os.path.join(lsst.utils.getPackageDir("ap_association"),
                                "data/association-flag-map.yaml")
        self._unpacker = UnpackApdbFlags(flag_map, "DiaSource")

        self.set_excluded_diaSource_flags(['base_PixelFlags_flag_bad',
                                           'base_PixelFlags_flag_suspect',
                                           'base_PixelFlags_flag_saturatedCenter',
                                           'base_PixelFlags_flag_interpolated',
                                           'base_PixelFlags_flag_interpolatedCenter',
                                           'base_PixelFlags_flag_edge',
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
            if not self._unpacker.flagExists(flag, columnName='flags'):
                raise ValueError(f"flag {flag} not included in DiaSource flags")

        self.diaSource_flags_exclude = flag_list

    def _make_flag_exclusion_query(self, query, table, flag_list, column_name='flags'):
        """Return an SQL where query that excludes sources with chosen flags.

        Parameters
        ----------
        flag_list : `list` [`str`]
            Flag names to exclude.
        query : `sqlalchemy.sql.Query`
            Query to include the where statement in.
        table : `sqlalchemy.schema.Table`
            Table containing the column to be queried.
        column_name : `str`, optional
            Name of flag column to query.

        Returns
        -------
        clause : `str`
            Clause to include in the SQL where statement.
        """

        bitmask = int(self._unpacker.makeFlagBitMask(flag_list, columnName=column_name))

        if bitmask == 0:
            warnings.warn(f"Flag bitmask is zero. Supplied flags: {flag_list}", RuntimeWarning)

        return query.where(table.columns[column_name].op("&")(bitmask) == 0)

    def load_sources_for_object(self, dia_object_id, exclude_flagged=False, limit=100000):
        """Load diaSources for a single diaObject.

        Parameters
        ----------
        dia_object_id : `int`
            Id of object to load sources for.
        exclude_flagged : `bool`, optional
            Exclude sources that have selected flags set.
            Use `set_DiaSource_exclude_flags` to configure which flags
            are excluded.
        limit : `int`
            Maximum number of rows to return.

        Returns
        -------
        data : `pandas.DataFrame`
            DiaSources for the specified diaObject.
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

    def load_source(self, id):
        """Load one diaSource.

        Parameters
        ----------
        id : `int`
            DiaSourceId to load data for.

        Returns
        -------
        data : `pandas.Series`
            The requested source.
        """
        table = self._tables["DiaSource"]
        query = table.select().where(table.columns["diaSourceId"] == id)
        with self.connection as connection:
            result = pd.read_sql_query(query, connection)
        self._fill_from_ccdVisitId(result)
        return result.iloc[0]

    def load_sources(self, exclude_flagged=False, limit=100000):
        """Load DiaSources.

        Parameters
        ----------
        exclude_flagged : `bool`, optional
            Exclude sources that have selected flags set.
            Use `set_DiaSource_exclude_flags` to configure which flags
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
        """Load one diaObject.

        Parameters
        ----------
        id : `int`
            DiaObjectId to load data for.

        Returns
        -------
        data : `pandas.Series`
            The requested object.
        """
        table = self._tables["DiaObject"]
        query = table.select().where(table.columns["validityStart"].isnot(None))
        query = query.where(table.columns["diaObjectId"] == id)
        with self.connection as connection:
            result = pd.read_sql_query(query, connection)
        return result.iloc[0]

    def load_objects(self, limit=100000):
        """Load all DiaObjects.

        Parameters
        ----------
        limit : `int`
            Maximum number of rows to return.

        Returns
        -------
        data : `pandas.DataFrame`
            All available diaObjects.
        """
        table = self._tables["DiaObject"]
        query = table.select().where(table.columns["validityStart"].isnot(None))
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
            DiaForcedSourceId to load data for.

        Returns
        -------
        data : `pandas.Series`
            The requested forced source.
        """
        table = self._tables["DiaForcedSource"]
        query = table.select().where(table.columns["diaForcedSourceId"] == id)
        with self.connection as connection:
            result = pd.read_sql_query(query, connection)
        self._fill_from_ccdVisitId(result)
        return result.iloc[0]

    def load_forced_sources(self, limit=100000):
        """Load all DiaForcedSources.

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

    def _fill_from_ccdVisitId(self, diaSource):
        """Expand the ccdVisitId value in the database.
        This method is temporary, until the CcdVisit table is filled out.

        Parameters
        ----------
        diaSource : `pandas.core.frame.DataFrame`
            Pandas dataframe with DIA Sources from an APDB; modified in-place.
        """
        # do nothing for an empty series
        if len(diaSource) == 0:
            return
        instrumentDataId = self._butler.registry.expandDataId(instrument=self._instrument)
        packer = self._butler.registry.dimensions.makePacker("visit_detector", instrumentDataId)
        dataId = packer.unpack(diaSource.ccdVisitId)
        diaSource['visit'] = dataId['visit']
        diaSource['detector'] = dataId['detector']
        diaSource['instrument'] = self._instrument


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
        self._engine = sqlalchemy.create_engine(f"sqlite:///{filename}")

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

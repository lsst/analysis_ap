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
import sqlite3
import warnings

import pandas as pd
import psycopg


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

    @property
    @contextlib.contextmanager
    @abc.abstractmethod
    def connection(self):
        """Context manager for database connections.

        Must yield an sql Connection object like ``psycopg.Connection``;
        whether the connection is closed after the context manager is closed
        is implementation dependent.
        """
        pass

    def load_sources_for_object(self, dia_object_id):
        """Load diaSources for a single diaObject.

        Parameters
        ----------
        dia_object_id : `int`
            Id of object to load sources for.

        Returns
        -------
        data : `pandas.DataFrame`
            DiaSources for the specified diaObject.
        """
        with self.connection as connection:
            order = 'ORDER BY "ccdVisitId", "diaSourceId"'
            query = ('SELECT * FROM "DiaSource"'
                     f' WHERE "DiaSource"."diaObjectId" = {dia_object_id}'
                     f' {order};')
            result = pd.read_sql_query(query, connection)
        self._fill_from_ccdVisitId(result)
        return result

    def load_sources(self, limit=100000):
        """Load all DiaSources.

        Parameters
        ----------
        limit : `int`
            Maximum number of rows to return.

        Returns
        -------
        data : `pandas.DataFrame`
            All available diaSources.
        """
        with self.connection as connection:
            order = 'ORDER BY "ccdVisitId", "diaSourceId"'
            limit_str = f"LIMIT {limit}" if limit is not None else ""
            query = f'SELECT * FROM "DiaSource" {order} {limit_str};'
            result = pd.read_sql_query(query, connection)
        self._fill_from_ccdVisitId(result)
        return result

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
        with self.connection as connection:
            order = 'ORDER BY "diaObjectId"'
            limit_str = f"LIMIT {limit}" if limit is not None else ""
            query = f'SELECT * FROM "DiaObject" {order} {limit_str};'
            result = pd.read_sql_query(query, connection)
        return result

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
        with self.connection as connection:
            order = 'ORDER BY "ccdVisitId", "diaSourceId"'
            limit_str = f"LIMIT {limit}" if limit is not None else ""
            query = f'SELECT * FROM "DiaForcedSource" {order} {limit_str};'
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
        self._connection = sqlite3.connect(filename)

    @property
    @contextlib.contextmanager
    def connection(self):
        yield self._connection


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

    @property
    @contextlib.contextmanager
    def connection(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")
            _connection = psycopg.connect(self._connection_string)
            cursor = _connection.cursor()
            cursor.execute(psycopg.sql.SQL("SET search_path TO {}").format(
                psycopg.sql.Identifier(self._namespace)))
            try:
                yield _connection
            finally:
                _connection.close()

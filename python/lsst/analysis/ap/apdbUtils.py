# This file is part of pipe_tasks.
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

"""Utility functions for connecting to, and manipulating the output from, APDB.
"""

__all__ = ["connectToApdb", "addTableMetadata"]

import sqlite3
import psycopg2


def connectToApdb(dbName, dbType='sqlite', schema=None):
    """Connect to an sqlite or postgres APDB.

    Parameters
    ----------
    dbName : `str`
        If dbType is sqlite, path to the APDB on lsst-devl.
        If dbType is postgres, name of the APDB on lsst-pg-devel1.
    dbType : `str`, optional
        Either 'sqlite' or 'postgres'
    schema : `str`, optional
        Required if dbType is postgres

    Returns
    -------
    connection : `psycopg2.connect` or `sqlite3.Connection`
        A connection object to a database instance, ready for queries
    """
    if dbType == 'sqlite':
        connection = sqlite3.connect(dbName)
    elif dbType == 'postgres':
        if schema is None:
            raise RuntimeError('Schema must be set for postgres APDB')
        host = 'lsst-pg-devel1.ncsa.illinois.edu'
        connection = psycopg2.connect(dbname=dbName,
                                      host=host,
                                      options=f'-c search_path={schema}')
    else:
        raise ValueError(f'dbType must be sqlite or postgres, not {dbType}')

    return connection


def addTableMetadata(sourceTable, butler, instrument='DECam'):
    """Add visit,detector,instrument columns to a DiaSource dataframe.

    Parameters
    ----------
    sourceTable : `pandas.core.frame.DataFrame`
        Pandas dataframe with DIA Sources from an APDB; modified in-place.
    butler : `lsst.daf.butler.Butler`
        Butler in the repository corresponding to the output of an ap_pipe run.
    instrument : `str`, optional
        Short name (e.g. "DECam") of instrument to make a dataId unpacker
        and to add to the table columns; supports any gen3 instrument.

    Notes
    -----
    This function should be removed once the visit/detector/instrument columns
    are available from the APDB schema, and populated with the correct values.
    """
    instrumentDataId = butler.registry.expandDataId(instrument=instrument)
    packer = butler.registry.dimensions.makePacker("visit_detector", instrumentDataId)
    dataId = packer.unpack(sourceTable.ccdVisitId)
    sourceTable['visit'] = dataId['visit']
    sourceTable['detector'] = dataId['detector']
    sourceTable['instrument'] = instrument

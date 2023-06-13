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

"""Utility functions for connecting to, and manipulating the output from, APDB.

All of the functions in this module are deprecated, and are currently being
retained for backwards compatibility with existing ap_pipe notebooks.
"""

__all__ = ["connectToApdb", "addTableMetadata", "loadSelectApdbSources",
           "loadExposures", "loadTables",
           "loadTablesByVisit", "loadTablesByBand", "makeSrcTableFlags",
           "loadAllApdbObjects", "loadAllApdbSources", "loadApdbSourcesByVisit",
           "loadApdbSourcesByBand"]

import functools
import operator
import os
from deprecated.sphinx import deprecated

import sqlite3

import psycopg2
from psycopg2 import sql
import pandas as pd

from lsst.ap.association import UnpackApdbFlags, TransformDiaSourceCatalogConfig
import lsst.daf.butler as dafButler


@deprecated(reason="This method is deprecated and will be removed once the "
                   "replacement API is in place.", version="v24", category=FutureWarning)
def connectToApdb(dbName, dbType='sqlite', schema=None,
                  user="rubin", host="usdf-prompt-processing-dev.slac.stanford.edu"):
    """Connect to an sqlite or postgres APDB.

    Parameters
    ----------
    dbName : `str`
        If dbType is "sqlite", path to the APDB on the host.
        If dbType is "postgres", name of the APDB on the DB host machine.
    dbType : `str`, optional
        Either "sqlite" or "postgres".
    schema : `str`, optional
        Required if dbType is "postgres".
    user : `str`, optional
        Username to connect to postgres database with; only used for
        ``dbType="postgres"``.
    host: `str`, optional
        Hostname of postgres database to connect to; only used for
        ``dbType="postgres"``.

    Returns
    -------
    connection : `psycopg2.connect` or `sqlite3.Connection`
        A connection object to a database instance, ready for queries
    """
    if dbType == "sqlite":
        connection = sqlite3.connect(dbName)
    elif dbType == "postgres":
        if schema is None:
            raise RuntimeError('Schema must be set for postgres APDB')
        connection = psycopg2.connect(dbname=dbName,
                                      host=host,
                                      user=user,
                                      )
        cursor = connection.cursor()
        cursor.execute(sql.SQL("SET search_path TO {}").format(
            sql.Identifier(schema)))
    else:
        raise ValueError(f"dbType must be sqlite or postgres, not {dbType}")

    return connection


@deprecated(reason="This method is deprecated and will be removed once the "
                   "replacement API is in place.", version="v24", category=FutureWarning)
def addTableMetadata(sourceTable, butler, instrument):
    """Add visit,detector,instrument columns to a DiaSource dataframe.

    Parameters
    ----------
    sourceTable : `pandas.core.frame.DataFrame`
        Pandas dataframe with DIA Sources from an APDB; modified in-place.
    butler : `lsst.daf.butler.Butler`
        Butler in the repository corresponding to the output of an ap_pipe run.
    instrument : `str`
        Short name (e.g. "DECam") of instrument to make a dataId unpacker
        and to add to the table columns; supports any gen3 instrument.

    Notes
    -----
    This function should be removed once the visit/detector/instrument columns
    are available from the APDB schema, and populated with the correct values.
    """
    if instrument is None:
        raise RuntimeError("Must specify instrument when expanding catalog metadata.")
    instrumentDataId = butler.registry.expandDataId(instrument=instrument)
    packer = butler.dimensions.makePacker("visit_detector", instrumentDataId)
    dataId = packer.unpack(sourceTable.ccdVisitId)
    sourceTable['visit'] = dataId['visit']
    sourceTable['detector'] = dataId['detector']
    sourceTable['instrument'] = instrument


@deprecated(reason="This method is deprecated and will be removed once the "
                   "replacement API is in place.", version="v24", category=FutureWarning)
def loadSelectApdbSources(dbName, diaObjectId, dbType='sqlite', schema=None):
    """Load select columns from DIASources for a single DIAObject
    from an APDB into a pandas dataframe.

    Parameters
    ----------
    dbName : `str`
        If dbType is sqlite, full filepath to the APDB on lsst-dev.
        If dbType is postgres, name of the APDB on lsst-pg-devel1.
    diaObjectId : `int`
        DIA Object for which we want to retrieve constituent DIA Sources.
    dbType : `str`, optional
        Either 'sqlite' or 'postgres'
    schema : `str`, optional
        Required if dbType is postgres

    Returns
    -------
    srcTable : `pandas.DataFrame`
        DIA Source Table including the columns hard-wired below.
    """
    connection = connectToApdb(dbName, dbType, schema)

    # Load data from the source table
    srcTable = pd.read_sql_query('select "diaSourceId", "diaObjectId", \
                                  "ra", "decl", "ccdVisitId", "filterName", \
                                  "midPointTai", "apFlux", "psFlux", "apFluxErr", \
                                  "psFluxErr", "totFlux", "totFluxErr", "snr", "x", "y", \
                                  "ixxPSF", "iyyPSF", "ixyPSF", "flags" from {0}; \
                                  where "diaObjectId" = {1};'.format('"DiaSource"', diaObjectId), connection)
    return srcTable


@deprecated(reason="This method is deprecated and will be removed once the "
                   "replacement API is in place.", version="v24", category=FutureWarning)
def loadExposures(butler, dataId, collections, diffName='deep'):
    """Load a science exposure, difference image, and warped template.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler in the repository corresponding to the output of an ap_pipe run.
    dataId : `dict`-like
        Gen3 data ID specifying at least instrument, visit, and detector.
    collections : `str` or `list`
        Gen3 collection or collections from which to load the exposures.
    diffName : `str`, optional
        Default is 'deep', but 'goodSeeing' may be needed instead.

    Returns
    -------
    science : `lsst.afw.Exposure`
        calexp corresponding to dataId and collections.
    difference : `lsst.afw.Exposure`
        differenceExp corresponding to dataId and collections.
    template : `lsst.afw.Exposure`
        warpedExp corresponding to dataId and collections.
    """
    science = butler.get('calexp', dataId=dataId, collections=collections)
    difference = butler.get(diffName + 'Diff_differenceExp', dataId=dataId, collections=collections)
    template = butler.get(diffName + 'Diff_templateExp', dataId=dataId, collections=collections)
    return science, difference, template


@deprecated(reason="This method is deprecated and will be removed once the "
                   "replacement API is in place.", version="v24", category=FutureWarning)
def loadTables(repo, dbName='association.db', isVerify=False, dbType='sqlite',
               badFlagList=['base_PixelFlags_flag_bad',
                            'base_PixelFlags_flag_suspect',
                            'base_PixelFlags_flag_saturatedCenter',
                            'base_PixelFlags_flag_interpolated',
                            'base_PixelFlags_flag_interpolatedCenter',
                            'base_PixelFlags_flag_edge',
                            ],
               instrument=None, schema=None, allCol=False):
    """Load DIA Object and DIA Source tables from an APDB.

    Parameters
    ----------
    repo : `str`
        Repository corresponding to the output of an ap_pipe run.
    dbName : `str`
        Name of APDB.
    isVerify : `bool`
        Is this an ap_verify run instead of an ap_pipe run?
        If True, the APDB is one level above repo on disk.
        If False, the APDB is in repo (default).
    dbType : `str`, optional
        Either 'sqlite' or 'postgres'
    badFlagList :  `list`
        Names of flags presumed to each indicate a DIA
        Source is garbage.
    instrument : `str`, either 'DECam' or 'HSC', default is 'DECam'
        Needed to properly add the "ccd" and "visit" columns
        to the sourceTable, and for all things gen3
    schema : `str`, optional
        Required if dbType is postgres
    allCol : `bool`, optional
        If True, load ALL columns of data

    Returns
    -------
    objTable : `pandas.core.frame.DataFrame`
        DIA Object table loaded from the APDB.
    srcTable : `pandas.core.frame.DataFrame`
        DIA Source table loaded from the APDB.
    goodObj : `pandas.core.frame.DataFrame`
        A subset of objTable containing only DIA Objects composed
    entirely of good DIA Sources.
    goodSrc : `pandas.core.frame.DataFrame`
        A subset of srcTable containing only good DIA Sources.
    """
    if dbType == 'sqlite' and not isVerify:  # APDB is in repo (default)
        dbPath = os.path.abspath(os.path.join(repo, dbName))
    elif dbType == 'sqlite' and isVerify:  # APDB is one level above repo (ap_verify sqlite case)
        repoUpOne = os.path.dirname(repo)
        dbPath = os.path.abspath(os.path.join(repoUpOne, dbName))
    elif dbType == 'postgres':
        dbPath = dbName
    else:
        raise ValueError('database type not understood')

    butler = dafButler.Butler(repo)

    objTable = loadAllApdbObjects(dbPath, dbType=dbType,
                                  schema=schema, allCol=allCol)
    srcTable = loadAllApdbSources(dbPath, dbType=dbType,
                                  schema=schema, allCol=allCol)
    addTableMetadata(srcTable, butler=butler, instrument=instrument)
    flagTable, srcTableFlags, flagFilter, \
        goodSrc, goodObj = makeSrcTableFlags(srcTable, objTable, badFlagList=badFlagList,
                                             instrument=instrument, repo=repo)
    return objTable, srcTable, goodObj, goodSrc


@deprecated(reason="This method is deprecated and will be removed once the "
                   "replacement API is in place.", version="v24", category=FutureWarning)
def loadTablesByVisit(repo, visit, dbName='association.db', isVerify=False,
                      dbType='sqlite',
                      badFlagList=['base_PixelFlags_flag_bad',
                                   'base_PixelFlags_flag_suspect',
                                   'base_PixelFlags_flag_saturatedCenter',
                                   'base_PixelFlags_flag_interpolated',
                                   'base_PixelFlags_flag_interpolatedCenter',
                                   'base_PixelFlags_flag_edge',
                                   ],
                      instrument=None, schema=None):
    """Load DIA Object and DIA Source tables from an APDB.

    Parameters
    ----------
    repo : `str`
        Repository corresponding to the output of an ap_pipe run.
    visit: `int`
        Visit number
    dbName : `str`
        Name of APDB.
    isVerify : `bool`
        Is this an ap_verify run instead of an ap_pipe run?
        If True, the APDB is one level above repo on disk.
        If False, the APDB is in repo (default).
    dbType : `str`, optional
        Either 'sqlite' or 'postgres'
    badFlagList :  `list`
        Names of flags presumed to each indicate a DIA Source is garbage.
    instrument : `str`, one of either 'DECam' or 'HSC', default is 'DECam'
        Needed to properly add the "ccd" and "visit" columns to
    the sourceTable, and for all things gen3
    schema : `str`, optional
        Required if dbType is postgres

    Returns
    -------
    objTable : `pandas.core.frame.DataFrame`
        DIA Object table loaded from the APDB.
    srcTable : `pandas.core.frame.DataFrame`
        DIA Source table loaded from the APDB.
    goodObj : `pandas.core.frame.DataFrame`
        A subset of objTable containing only DIA Objects composed entirely of
        good DIA Sources.
    goodSrc : `pandas.core.frame.DataFrame`
        A subset of srcTable containing only good DIA Sources.
    """

    if dbType == 'sqlite' and not isVerify:  # APDB is in repo (default)
        dbPath = os.path.abspath(os.path.join(repo, dbName))
    elif dbType == 'sqlite' and isVerify:  # APDB is one level above repo (ap_verify sqlite case)
        repoUpOne = os.path.dirname(repo)
        dbPath = os.path.abspath(os.path.join(repoUpOne, dbName))
    elif dbType == 'postgres':
        dbPath = dbName
    else:
        raise ValueError('database type not understood')

    butler = dafButler.Butler(repo)

    objTable = loadAllApdbObjects(dbPath, dbType=dbType, schema=schema)
    srcTable = loadApdbSourcesByVisit(dbPath, visit, dbType=dbType, schema=schema)
    addTableMetadata(srcTable, butler=butler, instrument=instrument)
    flagTable, srcTableFlags, flagFilter, \
        goodSrc, goodObj = makeSrcTableFlags(srcTable, objTable, badFlagList=badFlagList,
                                             instrument=instrument, repo=repo)
    return objTable, srcTable, goodObj, goodSrc


@deprecated(reason="This method is deprecated and will be removed once the "
                   "replacement API is in place.", version="v24", category=FutureWarning)
def loadTablesByBand(repo, band, dbName='association.db', isVerify=False,
                     dbType='sqlite',
                     badFlagList=['base_PixelFlags_flag_bad',
                                  'base_PixelFlags_flag_suspect',
                                  'base_PixelFlags_flag_saturatedCenter',
                                  'base_PixelFlags_flag_interpolated',
                                  'base_PixelFlags_flag_interpolatedCenter',
                                  'base_PixelFlags_flag_edge',
                                  ],
                     instrument=None, schema=None):
    """Load DIA Object and DIA Source tables from an APDB.

    Parameters
    ----------
    repo : `str`
        Repository corresponding to the output of an ap_pipe run.
    band: `str`
        Band to match against filterName in DB
    dbName : `str`
        Name of APDB.
    isVerify : `bool`
        Is this an ap_verify run instead of an ap_pipe run?
        If True, the APDB is one level above repo on disk.
        If False, the APDB is in repo (default).
    dbType : `str`, optional
        Either 'sqlite' or 'postgres'
    badFlagList :  `list`
        Names of flags presumed to each indicate a DIA Source is garbage.
    instrument : `str`, one of either 'DECam' or 'HSC', default is 'DECam'
        Needed to properly add the "ccd" and "visit" columns to the
        sourceTable, and for all things gen3.
    schema : `str`, optional
        Required if dbType is postgres

    Returns
    -------
    objTable : `pandas.core.frame.DataFrame`
        DIA Object table loaded from the APDB.
    srcTable : `pandas.core.frame.DataFrame`
        DIA Source table loaded from the APDB.
    goodObj : `pandas.core.frame.DataFrame`
        A subset of objTable containing only DIA Objects composed entirely of
        good DIA Sources.
    goodSrc : `pandas.core.frame.DataFrame`
        A subset of srcTable containing only good DIA Sources.
    """
    if dbType == 'sqlite' and not isVerify:  # APDB is in repo (default)
        dbPath = os.path.abspath(os.path.join(repo, dbName))
    # APDB is one level above repo (ap_verify sqlite case)
    elif dbType == 'sqlite' and isVerify:
        repoUpOne = os.path.dirname(repo)
        dbPath = os.path.abspath(os.path.join(repoUpOne, dbName))
    elif dbType == 'postgres':
        dbPath = dbName
    else:
        raise ValueError('database type not understood')

    butler = dafButler.Butler(repo)

    objTable = loadAllApdbObjects(dbPath, dbType=dbType, schema=schema)
    srcTable = loadApdbSourcesByBand(dbPath, band, dbType=dbType, schema=schema)
    addTableMetadata(srcTable, butler=butler, instrument=instrument)
    flagTable, srcTableFlags, flagFilter, \
        goodSrc, goodObj = makeSrcTableFlags(srcTable, objTable, badFlagList=badFlagList,
                                             instrument=instrument, repo=repo)
    return objTable, srcTable, goodObj, goodSrc


@deprecated(reason="This method is deprecated and will be removed once the "
                   "replacement API is in place.", version="v24", category=FutureWarning)
def makeSrcTableFlags(sourceTable, objectTable,
                      badFlagList=['base_PixelFlags_flag_bad',
                                   'base_PixelFlags_flag_suspect',
                                   'base_PixelFlags_flag_saturatedCenter',
                                   'base_PixelFlags_flag_interpolated',
                                   'base_PixelFlags_flag_interpolatedCenter',
                                   'base_PixelFlags_flag_edge',
                                   ],
                      instrument=None, repo=None):
    """Apply flag filters to a DIA Source and a DIA Object table.

    Parameters
    ----------
    sourceTable : `pandas.core.frame.DataFrame`
        Pandas dataframe with DIA Sources from an APDB.
    objectTable : `pandas.core.frame.DataFrame`
        Pandas dataframe with DIA Objects from the same APDB.
    badFlagList :  `list`
        Names of flags presumed to each indicate a DIA Source is garbage.
     instrument : `str`
        Default is 'DECam'
    repo : `str`
        Repository in which to load a butler

    Returns
    -------
    flagTable : `pandas.core.frame.DataFrame`
        Dataframe containing unpacked DIA Source flag values.
    sourceTableFlags : `pandas.core.frame.DataFrame`
        Dataframe resulting from from merging sourceTable with flagTable.
    flagFilter : `pandas.core.series.Series` of `bool`
        Single column of booleans of length len(sourceTable).
        The value of flagFilter is True if one or more flags
        in badFlagList is True.
    goodSrc : `pandas.core.frame.DataFrame`
        Dataframe containing only DIA Sources from sourceTable
        with no bad flags.
    goodObj : `pandas.core.frame.DataFrame`
        Dataframe containing only DIA Objects from objectTable
        entirely composed of DIA Sources with no bad flags.
    """
    butler = dafButler.Butler(repo)
    addTableMetadata(sourceTable, butler=butler, instrument=instrument)
    config = TransformDiaSourceCatalogConfig()
    unpacker = UnpackApdbFlags(config.flagMap, 'DiaSource')
    flagValues = unpacker.unpack(sourceTable['flags'], 'flags')
    flagTable = pd.DataFrame(flagValues, index=sourceTable.index)
    sourceTableFlags = pd.merge(sourceTable, flagTable, left_index=True, right_index=True)
    badFlags = [sourceTableFlags[flag] for flag in badFlagList]
    flagFilter = functools.reduce(operator.or_, badFlags)
    noFlagFilter = ~flagFilter
    goodSrc = sourceTableFlags.loc[noFlagFilter]
    goodObjIds = set(sourceTableFlags.loc[noFlagFilter, 'diaObjectId'])
    goodObj = objectTable.loc[objectTable['diaObjectId'].isin(goodObjIds)]
    return flagTable, sourceTableFlags, flagFilter, goodSrc, goodObj


@deprecated(reason="This method is deprecated and will be removed once the "
                   "replacement API is in place.", version="v24", category=FutureWarning)
def loadAllApdbObjects(dbName, dbType='sqlite', schema=None, allCol=False):
    """Load a subset of DIAObject columns from a APDB into a pandas dataframe.

    Parameters
    ----------
    dbName : `str`
        If dbType is sqlite, *full path* to the APDB on lsst-dev.
        If dbType is postgres, name of the APDB on lsst-pg-devel1.
    dbType : `str`, optional
        Either 'sqlite' or 'postgres'
    schema : `str`, optional
        Required if dbType is postgres
    allCol : `bool`, optional
        If True, load ALL columns of data

    Returns
    -------
    objTable : `pandas.DataFrame`
        DIA Object Table containing only objects with validityEnd NULL.
        Columns selected are presently hard-wired here.
    """
    connection = connectToApdb(dbName, dbType, schema)

    # Only get objects with validityEnd NULL as they are still valid
    if not allCol:
        objTable = pd.read_sql_query('select "diaObjectId", "ra", "decl", "nDiaSources", \
                                    "gPSFluxMean", "rPSFluxMean", "iPSFluxMean", \
                                    "zPSFluxMean", "yPSFluxMean", "validityEnd", "flags" from {0} \
                                    where "validityEnd" is NULL;'.format('"DiaObject"'), connection)
    else:
        objTable = pd.read_sql_query('select * from {0} where "validityEnd" is NULL; \
                                     '.format('"DiaObject"'), connection)
    return objTable


@deprecated(reason="This method is deprecated and will be removed once the "
                   "replacement API is in place.", version="v24", category=FutureWarning)
def loadAllApdbSources(dbName, dbType='sqlite', schema=None, allCol=False):
    """Load a subset of columns from all DIASources from an APDB
       into a pandas dataframe.

    Parameters
    ----------
    dbName : `str`
        If dbType is sqlite, full filepath to the APDB on lsst-dev.
        If dbType is postgres, name of the APDB on lsst-pg-devel1.
    dbType : `str`, optional
        Either 'sqlite' or 'postgres'
    schema : `str`, optional
        Required if dbType is postgres
    allCol : `bool`, optional
        If True, load ALL columns of data

    Returns
    -------
    srcTable : `pandas.DataFrame`
        DIA Source Table including the columns hard-wired below.
    """
    connection = connectToApdb(dbName, dbType, schema)

    # Load data from the source table
    if not allCol:
        srcTable = pd.read_sql_query('select "diaSourceId", "diaObjectId", \
                                    "ra", "decl", "ccdVisitId", \
                                    "midPointTai", "apFlux", "psFlux", "apFluxErr", \
                                    "psFluxErr", "totFlux", "totFluxErr", "snr", "x", "y", \
                                    "ixxPSF", "iyyPSF", "ixyPSF", "flags", "filterName" from {0}; \
                                    '.format('"DiaSource"'), connection)
    else:
        srcTable = pd.read_sql_query('select * from {0}; \
                                     '.format('"DiaSource"'), connection)
    return srcTable


@deprecated(reason="This method is deprecated and will be removed once the "
                   "replacement API is in place.", version="v24", category=FutureWarning)
def loadAllApdbForcedSources(dbName, dbType='sqlite', schema=None, allCol=False):
    """Load columns from all ForcedDiaSources from a APDB
    into a pandas dataframe.

    Parameters
    ----------
    dbName : `str`
        If dbType is sqlite, full filepath to the APDB.
        If dbType is postgres, name of the APDB on the host.
    dbType : `str`, optional
        Either 'sqlite' or 'postgres'
    schema : `str`, optional
        Required if dbType is postgres
    allCol : `bool`, optional
        If True, load ALL columns of data

    Returns
    -------
    srcTable : `pandas.DataFrame`
        DIA Source Table including the columns hard-wired below,
        or all columns if allCol is set to True.
    """
    connection = connectToApdb(dbName, dbType, schema)

    # Load data from the forced source table
    if not allCol:
        forcedSrcTable = pd.read_sql_query('select "diaSourceId", "diaObjectId", \
                                        "ra", "decl", "ccdVisitId", \
                                        "midPointTai", "apFlux", "psFlux", "apFluxErr", \
                                        "psFluxErr", "totFlux", "totFluxErr", "snr", "x", "y", \
                                        "ixxPSF", "iyyPSF", "ixyPSF", "flags", "filterName" from {0}; \
                                        '.format('"DiaForcedSource"'), connection)
    else:
        forcedSrcTable = pd.read_sql_query('select * from {0}; \
                                        '.format('"DiaForcedSource"'), connection)
    return forcedSrcTable


@deprecated(reason="This method is deprecated and will be removed once the "
                   "replacement API is in place.", version="v24", category=FutureWarning)
def loadApdbSourcesByVisit(dbName, visit, dbType='sqlite', schema=None):
    """Load a subset of columns from all DIASources from an APDB
       into a pandas dataframe.

    Parameters
    ----------
    dbName : `str`
        If dbType is sqlite, full filepath to the APDB on lsst-dev.
        If dbType is postgres, name of the APDB on lsst-pg-devel1.
    dbType : `str`, optional
        Either 'sqlite' or 'postgres'
    schema : `str`, optional
        Required if dbType is postgres
    visit : `int`
        Visit number for loading objects

    Returns
    -------
    srcTable : `pandas.DataFrame`
        DIA Source Table including the columns hard-wired below.
    """
    connection = connectToApdb(dbName, dbType, schema)
    # Load data from the source table
    srcTable = pd.read_sql_query('select "diaSourceId", "diaObjectId", \
                                  "ra", "decl", "ccdVisitId", \
                                  "midPointTai", "apFlux", "psFlux", "apFluxErr", \
                                  "psFluxErr", "totFlux", "totFluxErr", "snr", "x", "y", \
                                  "ixxPSF", "iyyPSF", "ixyPSF", "flags", "filterName" from {0} \
                                   where CAST("ccdVisitId" as text) like {1} ; \
                                  '.format('"DiaSource"', "'"+str(int(visit))+"%'"), connection)
    return srcTable


@deprecated(reason="This method is deprecated and will be removed once the "
                   "replacement API is in place.", version="v24", category=FutureWarning)
def loadApdbSourcesByBand(dbName, band, dbType='sqlite', schema=None):
    """Load a subset of columns from all DIASources from an APDB
       into a pandas dataframe.

    Parameters
    ----------
    dbName : `str`
        If dbType is sqlite, full filepath to the APDB on lsst-dev.
        If dbType is postgres, name of the APDB on lsst-pg-devel1.
    dbType : `str`, optional
        Either 'sqlite' or 'postgres'
    schema : `str`, optional
        Required if dbType is postgres
    band : `str`
        Band for loading objects (matched against filterName)

    Returns
    -------
    srcTable : `pandas.DataFrame`
        DIA Source Table including the columns hard-wired below.
    """
    connection = connectToApdb(dbName, dbType, schema)
    # Load data from the source table
    srcTable = pd.read_sql_query('select "diaSourceId", "diaObjectId", \
                                  "ra", "decl", "ccdVisitId", \
                                  "midPointTai", "apFlux", "psFlux", \
                                  "apFluxErr", "psFluxErr", "totFlux", \
                                  "totFluxErr", "snr", "x", "y", "ixxPSF", \
                                  "iyyPSF", "ixyPSF", "flags", "filterName" \
                                  from {0} where "filterName" = {1} ; \
                                  '.format('"DiaSource"', "'"+band+"'"),
                                 connection)
    return srcTable

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

__all__ = ["load_exposures", "makeSrcTableFlags"]

import functools
import operator
from deprecated.sphinx import deprecated

import pandas as pd

from lsst.ap.association import UnpackApdbFlags, TransformDiaSourceCatalogConfig
import lsst.daf.butler as dafButler


# TODO: this can move somewhere outside of legacy, as it is handy.
def load_exposures(butler, dataId, diffName='deep'):
    """Load a science exposure, difference image, and warped template.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler in the repository corresponding to the output of an ap_pipe
        run; create it with the collections you wish to load images from.
    dataId : `dict`
        Gen3 data ID specifying at least instrument, visit, and detector.
    diffName : `str`, optional
        Default is 'deep', but 'goodSeeing' may be needed instead.

    Returns
    -------
    science : `lsst.afw.Exposure`
        science exposure to load for this dataId.
    difference : `lsst.afw.Exposure`
        difference exposure to load for this dataId.
    template : `lsst.afw.Exposure`
        template exposure to load for this dataId.
    """
    science = butler.get('calexp', dataId=dataId)
    difference = butler.get(diffName + 'Diff_differenceExp', dataId=dataId)
    template = butler.get(diffName + 'Diff_templateExp', dataId=dataId)
    return science, difference, template


# TODO: ApdbQuery needs a flag bits->named columns method, or a free function.
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


# TODO: ApdbQuery needs a select (in particular for visit and band) interface.
def loadApdbSourcesByVisit(dbName, visit, dbType='sqlite', schema=None):
    pass
def loadApdbSourcesByBand(dbName, band, dbType='sqlite', schema=None):
    pass

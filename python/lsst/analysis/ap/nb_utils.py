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

__all__ = ["make_simbad_link", "compare_sources"]

import astropy.coordinates as coord
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astroquery.simbad import Simbad
import astropy.units as u
import os
import pandas as pd
import numpy as np

from lsst.analysis.ap import plotImageSubtractionCutouts
from IPython.display import display, Image


def make_simbad_link(ra, dec, radius_arcsec=3.0):
    """Search Simbad for associated sources within a 3 arcsecond region.

    Parameters
    ----------
    ra : 'float'
        Ra from source.

    dec : 'float'
        Dec from source.

    radius_arcsec : 'float'
        Search radius submitted to Simbad in arcseconds.
        Default radius is 3 arcseconds.

    Returns
    -------
    results_table : `astropy.table.table.Table`
        A table of Simbad search results.
    """
    # IPython is not in the base conda env, so hide the import here.
    from IPython.display import display, Markdown

    search_results = f"http://simbad.cds.unistra.fr/simbad/sim-coo?Coord={ra}+{dec}" \
                     f"&CooFrame=FK5&CooEpoch=2000&CooEqui=2000&CooDefinedFrames=none&Radius=" \
                     f"{radius_arcsec}&Radius.unit=arcsec&submit=submit+query&CoordList="
    display(Markdown(f"[Link to Simbad search]({search_results})"))

    source_coords = coord.SkyCoord(ra, dec, frame="icrs", unit=(u.deg, u.deg))
    customSimbad = Simbad()
    customSimbad.TIMEOUT = 600
    customSimbad.add_votable_fields("otype(V)")
    results_table = customSimbad.query_region(
        source_coords, radius=radius_arcsec*u.arcsecond
    )

    if results_table is not None:

        return results_table

    else:
        print(f"No matched sources within {radius_arcsec} arcseconds.")

        return None


def compare_sources(butler1, butler2, query1, query2,
                    bad_flag_list=None, match_radius=0.1,
                    make_cutouts=False, display_cutouts=False,
                    cutout_path1=None, cutout_path2=None,
                    cutout_config1=None, cutout_config2=None,
                    njobs=0):
    """Compare two APDB datasets by extracting unassociated sources,
    spatially crossmatching, and plotting cutouts of the differences.

    Parameters
    ----------
    butler1 : `lsst.daf.butler`
        Initialized Butler repo containing the first dataset.
        Could be the same as butler2 but should be initialized with the
        appropriate collection name for cutout generation, if doing that.
    butler2 : `lsst.daf.butler`
        Initialized Butler repo containing the second dataset.
        Could be the same as butler1 but should be initialized with the
        appropriate collection name for cutout generation, if doing that.
    query1 : `lsst.analysis.ap.DbQuery`
        DbQuery to first APDB (postgresql or slite file;
        NOT created in this function).
    query2 : `lsst.analysis.ap.DbQuery`
        DbQuery to second APDB (postgresql or slite file;
        NOT created in this function).
    bad_flag_list : `list`, optional
        List of bad flags to exclude (applied to both query1 and query2).
        Omit list to skip filtering.
    match_radius : `double`
        Maximum allowable distance in arcsec between an object in
        data1 and data2.
    make_cutouts : `bool`, optional
        Generate cutouts for sources unique to each dataset; default is False.
    display_cutouts: `bool`, optional
        Display cutouts for sources present in only one of the DBs to the
        screen; default is False.
    cutout_path1 : `str`, optional
        Base path to store cutouts for sources unique to dataset1.
        Must be supplied if make_cutouts is True.
    cutout_path2 : `str`, optional
        Base path to store cutouts for sources unique to dataset2.
        Must be supplied if make_cutouts is True.
    cutout_config1, cutout_config2 : `dict` [`str`], optional
        Config overrides to apply to cutout plotter for the datasets.
        See `~plotImageSubtractionCutouts.PlotImageSubtractionCutoutsConfig`
        for available options.
    njobs : `int`, optional
        Number of parallel processes for plotImageSubtractionCutouts.

    Returns
    -------
    unique1 : `pandas.DataFrame`
        Data frame of sources only found in the first dataset.
    unique2 : `pandas.DataFrame`
        Data frame of sources only found in the second dataset.
    matched : `pandas.DataFrame`
        Data frame of matched sources; actually only the sources from
        the first dataset but with a new column pointing to the
        diaSourceId of the match in the decond dataset.
    """

    if make_cutouts and (cutout_path1 is None or cutout_path2 is None):
        errstr = ('You must supply a value for `cutout_path1` and `cutout_path2` if `make_cutouts` is True.')
        raise ValueError(errstr)

    if bad_flag_list is not None:
        query1.set_excluded_diaSource_flags(bad_flag_list)
        query2.set_excluded_diaSource_flags(bad_flag_list)
    goodSrc1 = query1.load_sources(exclude_flagged=True)
    goodSrc2 = query2.load_sources(exclude_flagged=True)

    if 'reliability' not in goodSrc1.columns:
        goodSrc1['reliability'] = None
    if 'reliability' not in goodSrc2.columns:
        goodSrc2['reliability'] = None

    """
    We assume that the runs to be compared here will always be
    over the same dataset with the same visit and detector numbers.
    Then we can enforce the matching on that level, too.
    We could relax that assumption later, though it would require
    removing the visit, detector loop.
    """

    visit_det = set(zip(goodSrc1['visit'], goodSrc1['detector']))
    src1 = []
    src2 = []
    matches = []
    indices = {}
    seps = {}

    for visit, detector in visit_det:
        mask1 = (goodSrc1['visit'] == visit) &\
            (goodSrc1['detector'] == detector)
        mask2 = (goodSrc2['visit'] == visit) &\
            (goodSrc2['detector'] == detector)
        gs1 = goodSrc1[mask1].copy()
        gs2 = goodSrc2[mask2].copy()

        coords1 = SkyCoord(ra=gs1['ra'].values*u.degree,
                           dec=gs1['dec'].values*u.degree)
        coords2 = SkyCoord(ra=gs2['ra'].values*u.degree,
                           dec=gs2['dec'].values*u.degree)

        index, sep, _ = match_coordinates_sky(coords1, coords2)
        indices[(visit, detector)] = index
        seps[(visit, detector)] = sep
        gs1['xmatch_dist_arcsec'] = sep.to_value(u.arcsecond)
        gs1['src2_diaSourceId'] = gs2['diaSourceId'].values.astype(np.int64)[index]

        # Set the match ID to 0 if the distance is above threshold
        gs1.loc[(gs1['xmatch_dist_arcsec'] > match_radius),
                ['src2_diaSourceId']] = 0

        # get the DiaSources in dataset 2 not matched to something in dataset 1
        uniqueid2 = set(gs2['diaSourceId']) - set(gs1['src2_diaSourceId'])
        unique2 = gs2[gs2['diaSourceId'].isin(uniqueid2)]

        unique1 = gs1[(gs1['src2_diaSourceId'] == 0)]

        withmatch = gs1[(gs1['src2_diaSourceId'] > 0)]

        src1.append(unique1)
        src2.append(unique2)
        matches.append(withmatch)

    # Out of the loop, concatenate everything together.
    unique1 = pd.concat(src1)
    unique2 = pd.concat(src2)
    matched = pd.concat(matches)

    print("{} matched sources; {} unique to set 1; {} unique to set 2.".format(
        len(matched), len(unique1), len(unique2)))

    # Decide if we are doing anything with cutouts or not. If not, just skip.
    if make_cutouts:
        # Make paths if they don't exist.
        if not os.path.exists(cutout_path1):
            os.makedirs(cutout_path1)
        if not os.path.exists(cutout_path2):
            os.makedirs(cutout_path2)

        # Make cutouts if they don't already exist
        config1 = plotImageSubtractionCutouts.PlotImageSubtractionCutoutsConfig()
        config2 = plotImageSubtractionCutouts.PlotImageSubtractionCutoutsConfig()
        # default to flat directories for ease of use
        config1.chunk_size = None
        config2.chunk_size = None
        # apply user-specified overrides
        if cutout_config1 is not None:
            config1.update(**cutout_config1)
        if cutout_config2 is not None:
            config2.update(**cutout_config2)

        cpath1 = plotImageSubtractionCutouts.CutoutPath(cutout_path1,
                                                        chunk_size=config1.chunk_size)
        cpath2 = plotImageSubtractionCutouts.CutoutPath(cutout_path2,
                                                        chunk_size=config2.chunk_size)

        plotter1 = plotImageSubtractionCutouts.PlotImageSubtractionCutoutsTask(
            output_path=cutout_path1, config=config1)
        plotter2 = plotImageSubtractionCutouts.PlotImageSubtractionCutoutsTask(
            output_path=cutout_path2, config=config2)

        # First figure out which cutouts already exist at the output path
        unique1['pathexists'] = False
        for i in range(len(unique1)):
            dId = unique1.iloc[i]['diaSourceId']
            idx = unique1.index[i]
            unique1.at[idx, 'pathexists'] = os.path.exists(cpath1(dId))
        pathchk1 = unique1.loc[~unique1['pathexists']]

        unique2['pathexists'] = False
        for i in range(len(unique2)):
            dId = unique2.iloc[i]['diaSourceId']
            idx = unique2.index[i]
            unique2.at[idx, 'pathexists'] = os.path.exists(cpath2(dId))
        pathchk2 = unique2.loc[~unique2['pathexists']]

        # Only write those that don't exist yet
        plotter1.write_images(pathchk1, butler=butler1, njobs=njobs)
        plotter2.write_images(pathchk2, butler=butler2, njobs=njobs)

        if display_cutouts:
            for isrc in unique1.itertuples():
                fpath = cpath1(int(isrc.diaSourceId))
                print('Unique to dataset 1: {}'.format(int(isrc.diaSourceId)))
                display(Image(filename=fpath))

            for isrc in unique2.itertuples():
                fpath = cpath2(int(isrc.diaSourceId))
                print('Unique to dataset 2: {}'.format(int(isrc.diaSourceId)))
                display(Image(filename=fpath))

        # drop pathexists columns to return to original dataframe shape
        _ = unique1.pop('pathexists')
        _ = unique2.pop('pathexists')

    return unique1, unique2, matched

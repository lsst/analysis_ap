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

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.patches as patches
import pandas as pd
from deprecated.sphinx import deprecated

from astropy import units as u
from astropy.visualization import ZScaleInterval, SqrtStretch, ImageNormalize

import lsst.daf.butler as dafButler
import lsst.geom
from lsst.ap.association import UnpackApdbFlags, TransformDiaSourceCatalogConfig
import lsst.afw.cameraGeom as cameraGeom
from lsst.obs.decam import DarkEnergyCamera
from lsst.pipe.base import Struct

'''All of the functions herein are deprecated legacy functions.
In the comments, each has an associated ticket, and once those tickets
are complete so the plotting functionality is elsewhere, the legacy code
should be removed.
'''


# TODO: DM-43095, to plotUtils
@deprecated(reason="This function is deprecated and will be removed soon.",
            version="v27", category=FutureWarning)
def getPatchConstituents(repo, band='g', printConstituents=False, verbose=False,
                         tractIndex=0, instrument='DECam', collections=[],
                         skymapName='', coaddName='deepCoadd'):
    """Learn which patches have data in them, and what dataIds (i.e.,
    visits/exposures and detectors) comprise each patch.

    Parameters
    ----------
    repo : `str`
        The path to the data repository.
    band : `str`, optional
        The filter name of the data to select from the repository.
    printConstituents : `bool`, optional
        Select whether to print detailed information on each
            patch that has data.
    verbose : `bool`, optional
        Select whether to print all the patches with and without data.
    tractIndex : `int`, optional
        Tract index, default 0
    instrument : `str`, optional
        Default is 'DECam', used with gen3 butler only
    collections : `list` or `str`, optional
        Must be provided for gen3 to load the camera properly
    skymapName : `str`, optional
        Must be provided for gen3 to load the skymap. e.g., 'hsc_rings_v1'
    coaddName : `str`, optional
        Type of coadd, default `deepCoadd` (you might want `goodSeeingCoadd`)

    Returns
    -------
    dataPatchList : `list` of `str`
        The patches containing coadds in the repository.
    constituentList : `list` of `numpy.ndarray`
        The visit numbers of the calexps that contributed to each patch.
    constituentCountList : `list` of `int`
        The number of visits that contributed to each patch.
    """
    if not collections:
        raise ValueError('One or more collections is required in gen3.')
    if not skymapName:
        raise ValueError('A skymap is required.')
    if not instrument:
        raise ValueError('An instrument is required.')
    butler = dafButler.Butler(repo, collections=collections)
    skymap = butler.get('skyMap', instrument=instrument, collections='skymaps', skymap=skymapName)
    tract = skymap.generateTract(tractIndex)
    everyPatchList = [tract.getSequentialPatchIndex(tract[idx]) for idx in range(len(tract))]
    dataPatchList = []
    constituentList = []
    constituentCountList = []
    for patch in everyPatchList:
        try:
            dataIdPatch = {'band': band, 'tract': tractIndex,
                           'patch': int(patch), 'instrument': instrument,
                           'skymap': skymapName}
            coadd_test = butler.get(coaddName, dataId=dataIdPatch)
        except (LookupError):
            if verbose:
                print('No data in this patch', patch)
            continue
        else:
            if verbose:
                print('This patch has a coadd', patch)
            constituent = coadd_test.getInfo().getCoaddInputs().ccds['visit']
            constituentCount = len(constituent)
            if printConstituents:
                print(patch, constituentCount)
            dataPatchList.append(patch)
            constituentList.append(constituent)
            constituentCountList.append(constituentCount)
    return dataPatchList, constituentList, constituentCountList


# TODO: DM-43095, to plotUtils
@deprecated(reason="This function is deprecated and will be removed soon.",
            version="v27", category=FutureWarning)
def bboxToRaDec(bbox, wcs):
    """Get the corners of a BBox and convert them to lists of RA and Dec.

    Parameters
    ----------
    bbox : `lsst.geom.Box`
        The bounding box to determine coordinates for.
    wcs : `lsst.afw.geom.SkyWcs`
        The WCS to use to convert pixel to sky coordinates.

    Returns
    -------
    ra, dec : `tuple` of `float`
        The Right Ascension and Declination of the corners of the BBox.
    """
    skyCorners = [wcs.pixelToSky(pixPos.x, pixPos.y) for pixPos in bbox.getCorners()]
    ra = [corner.getRa().asDegrees() for corner in skyCorners]
    dec = [corner.getDec().asDegrees() for corner in skyCorners]
    return ra, dec


# TODO: DM-43095, to plotUtils
@deprecated(reason="This function is deprecated and will be removed soon.",
            version="v27", category=FutureWarning)
def getRaDecMinMaxPatchList(patchList, tractInfo, pad=0.0, nDecimals=4, raMin=360.0, raMax=0.0,
                            decMin=90.0, decMax=-90.0):
    """Find the max and min RA and DEC (deg) boundaries
       encompassed in the patchList
    Parameters
    ----------
    patchList : `list` of `str`
       List of patch IDs.
    tractInfo : `lsst.skymap.tractInfo.ExplicitTractInfo`
       Tract information associated with the patches in patchList
    pad : `float`
       Pad the boundary by pad degrees
    nDecimals : `int`
       Round coordinates to this number of decimal places
    raMin, raMax : `float`
       Initiate minimum[maximum] RA determination at raMin[raMax] (deg)
    decMin, decMax : `float`
       Initiate minimum[maximum] DEC determination at decMin[decMax] (deg)

    Returns
    -------
    `lsst.pipe.base.Struct`
       Contains the ra and dec min and max values for the patchList provided
    """
    for ip, patch in enumerate(tractInfo):
        patchWithData = tractInfo.getSequentialPatchIndex(patch)
        if patchWithData in patchList:
            raPatch, decPatch = bboxToRaDec(patch.getOuterBBox(), tractInfo.getWcs())
            raMin = min(np.round(min(raPatch) - pad, nDecimals), raMin)
            raMax = max(np.round(max(raPatch) + pad, nDecimals), raMax)
            decMin = min(np.round(min(decPatch) - pad, nDecimals), decMin)
            decMax = max(np.round(max(decPatch) + pad, nDecimals), decMax)
    return Struct(
        raMin=raMin,
        raMax=raMax,
        decMin=decMin,
        decMax=decMax,
    )


# TODO: DM-43095, to analysis_tools following DM-41531 (diaSource flags)
@deprecated(reason="This function is deprecated and will be removed soon.",
            version="v27", category=FutureWarning)
def plotDiaObjectHistogram(objects, row_mask,
                           label1='All Objects',
                           label2='Filtered Objects',
                           title=''):
    """Create a histogram showing how many DIA Sources comprise the DiaObjects.

    Parameters
    ----------
    objects : `pandas.DataFrame`
        DiaObject Table.
    row_mask : `array` [`bool`]
        Mask array for objects to plot filtered objects.
    label1 : `str`
        Legend label for the first DIA Object Table.
    label2 : `str`
        Legend label for the second (filtered) DIA Object Table.
    title : `str`
        Title for the plot, optional.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Histogram of DIA Objects showing number of constituent DIA Sources.
    """
    fig = plt.figure()
    plt.xlabel('Number of Sources per Object', size=16)
    plt.ylabel('Object count', size=16)
    plt.ylim(0.7, 1e5)
    plt.yscale('log')
    binMax = np.max(objects['nDiaSources'].values)
    plt.hist(objects['nDiaSources'].values, bins=np.arange(0, binMax),
             color='#2979c1', label=label1)
    plt.hist(objects.loc[row_mask, 'nDiaSources'].values, bins=np.arange(0, binMax),
             color='#bee7F5', label=label2)
    plt.legend(frameon=False, fontsize=16)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title(title)
    return fig


# TODO: delete this as part of DM-43201
@deprecated(reason="This function is deprecated and will be removed soon.",
            version="v27", category=FutureWarning)
def plotDiaObjectsOnSky(title, objects, row_mask):
    """Create a plot of filtered DIAObjects on the sky.

    Parameters
    ----------
    title : `str`
        Title to give to the plots.
    objects : `pandas.DataFrame`
        DIA Object Table.
    row_mask : `pandas.Series` of `bool`
        Filter applied to create a subset (e.g., quality cut) from objects.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Scatter plot of the DIAObjects on the sky with RA, Dec axes.
        Point sizes and colors correspond to the number of sources per object.
    """
    fig = plt.figure(facecolor='white', figsize=(10, 8))

    ax = fig.add_subplot(111)
    plot = ax.scatter(objects.loc[row_mask, 'ra'],
                      objects.loc[row_mask, 'decl'], marker='.', lw=0,
                      s=objects.loc[row_mask, 'nDiaSources']*8,
                      c=objects.loc[row_mask, 'nDiaSources'],
                      alpha=0.7, cmap='viridis', linewidth=0.5, edgecolor='k')
    plt.xlabel('RA (deg)', size=16)
    plt.ylabel('Dec (deg)', size=16)
    plt.title(title, size=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_xaxis()  # RA should increase to the left
    cb = fig.colorbar(plot, orientation='horizontal')
    cb.set_label('Number of Sources per Object', size=16)
    cb.set_clim(np.min(objects.loc[row_mask, 'nDiaSources']),
                np.max(objects.loc[row_mask, 'nDiaSources']))
    cb.solids.set_edgecolor("face")
    return fig


# TODO: DM-43095, to plotUtils
@deprecated(reason="This function is deprecated and will be removed soon.",
            version="v27", category=FutureWarning)
def plot2axes(x1, y1, x2, y2, xlabel, ylabel1, ylabel2, y1err=None, y2err=None, title=''):
    """Generic plot framework for showing one y-variable on the left
    and another on the right. The x-axis is shared on the bottom.

    Parameters
    ----------
    x1 : `list` or `array`
    y1 : `list` or `array`
    x2 : `list` or `array`
    y2 : `list` or `array`
    xlabel : `str`
        Label for shared x axis
    ylabel1 : `str`
        Label for y1 axis
    ylabel2 : `str`
        Label for y2 axis
    title : `str`
        Title for the plot, optional.
    """
    fig, ax1 = plt.subplots(figsize=(9, 3))
    color = 'C1'
    if y1err is not None:
        ax1.errorbar(x1, y1, yerr=y1err, color=color, marker='o', ls=':')
    else:
        ax1.plot(x1, y1, color=color, marker='o', ls=':')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    for label in ax1.get_xticklabels():
        label.set_ha("right")
        label.set_rotation(45)
        label.set_size("smaller")
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'C0'
    ax2.set_ylabel(ylabel2, color=color)  # we already handled the x-label above
    if y2err is not None:
        ax2.errorbar(x2, y2, yerr=y2err, color=color, marker='o', ls=':')
    else:
        ax2.plot(x2, y2, color=color, marker='o', ls=':')
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title(title)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped


# TODO: DM-43095, update the interface to take an instantiated butler,
# remove pandas, and add to an appropriate visit-level (nightly?) AP pipeline.
@deprecated(reason="This function is deprecated and will be removed soon.",
            version="v27", category=FutureWarning)
def plotDiaSourcesPerVisit(repo, sourceTable, title='',
                           instrument='DECam', collections=[]):
    """Plot DIA Sources per visit.

    The plot will have two y-axes: number of DIA Sources per square degree and
    median FWHM per ixx or iyy in pixels.

    Parameters
    ----------
    repo : `str`
        Butler repository corresponding to the output of an ap_pipe run.
    sourceTable : `pandas.core.frame.DataFrame`
        Pandas dataframe with DIA Sources from an APDB.
    title : `str`
        Title for the plot, optional.
    instrument : `str`, optional
        Default is 'DECam'
    collections : `list` or `str`
        Must be provided to load the camera properly
    """
    ccdArea, visitArea = getCcdAndVisitSizeOnSky(repo, sourceTable, instrument, collections)
    traceRadius = np.sqrt(0.5) * np.sqrt(sourceTable.ixxPSF + sourceTable.iyyPSF)
    sourceTable['seeing'] = 2*np.sqrt(2*np.log(2)) * traceRadius
    visitGroup = sourceTable.groupby('visit')
    plot2axes(visitGroup.visit.first().values,
              visitGroup.detector.count().values/visitArea,
              visitGroup.visit.first().values,
              visitGroup.seeing.median().values,
              'Visit',
              'Number of DIA Sources (per sq. deg.)',
              'Median FWHM per ixx/iyy (pixels)',
              title=title)


# TODO: DM-43095, update the interface to take an instantiated butler,
# remove pandas, use astropy times, and add to an appropriate visit-level
# AP pipeline.
@deprecated(reason="This function is deprecated and will be removed soon.",
            version="v27", category=FutureWarning)
def plotDiaSourcesPerNight(sourceTable, title=''):
    """Plot DIA Sources per night.

    The plot will have two y-axes: mean number of DIA Sources per visit and
    number of visits per night.

    Parameters
    ----------
    sourceTable : `pandas.core.frame.DataFrame`
        Pandas dataframe with DIA Sources from an APDB.
        NOT a view into or slice of a dataframe!
    title : `str`
        Title for the plot, optional.
    """
    date_times = pd.to_datetime(sourceTable['midPointTai'],
                                unit='D',
                                origin=pd.Timestamp('1858-11-17'))
    sourceTable['date'] = date_times.dt.date
    night_count = sourceTable.groupby(['date', 'visit']).count()
    visits_per_night = night_count.groupby('date').count()
    pervisit_per_night = night_count.groupby('date').mean()
    pervisit_per_night_std = night_count.groupby('date').std()
    pervisit_per_night_err = pervisit_per_night_std['x']/np.sqrt(visits_per_night['x'])
    plot2axes(visits_per_night.index,
              pervisit_per_night['x'],
              visits_per_night.index,
              visits_per_night['x'],
              'Night',
              'Mean DIA Sources per visit',
              'Number of Visits',
              y1err=pervisit_per_night_err,
              title=title)
    plt.ylim(0, np.max(visits_per_night['x'].values + 1))


# TODO: DM-43095, to plotUtils
@deprecated(reason="This function is deprecated and will be removed soon.",
            version="v27", category=FutureWarning)
def ccd2focalPlane(x, y, ccd, camera):
    """Retrieve focal plane coordinates.

    Parameters
    ----------
    x : `int` or `float`
        X-coordinate from ccd bbox.
    y : `int` or `float`
        Y-coordinate from ccd bbox.
    ccd : `int`, or can be cast as int
        The ccd being considered.
    camera : `lsst.afw.cameraGeom.Camera`
        Camera obtained from the butler.

    Returns
    -------
    point[0] : `int` or `float`
        X-coordinate in focal plane.
    point[1] : `int` or `float`
        Y-coordinate in focal plane.
    """
    detector = camera[int(ccd)]
    point = detector.transform(lsst.geom.Point2D(x, y),
                               cameraGeom.PIXELS, cameraGeom.FOCAL_PLANE)
    return point[0], point[1]


# TODO: DM-43095, to plotUtils, without pandas
@deprecated(reason="This function is deprecated and will be removed soon.",
            version="v27", category=FutureWarning)
def getCcdCorners(butler, sourceTable, instrument='DECam', collections=[]):
    """Get corner coordinates for a range of ccds.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler in the repository corresponding to the output of an ap_pipe run.
    sourceTable : `pandas.core.frame.DataFrame`
        Pandas dataframe with DIA Sources from an APDB.
    instrument : `str`, optional
        Default is 'DECam'
    collections : `list` or `str`
        Must be provided to load the camera properly

    Returns
    -------
    corners : `pandas.core.frame.DataFrame`
        Dataframe containing focal plane coordinates for all the ccd corners.
    """
    cornerList = []
    visits = np.unique(sourceTable['visit'])
    ccds = np.unique(sourceTable['detector'])
    ccdMin = int(np.min(ccds))
    ccdMax = int(np.max(ccds))
    visit = int(visits[0])  # shouldn't matter what visit we use
    for ccd in range(ccdMin, ccdMax+1):
        try:
            bbox = butler.get('calexp.bbox', collections=collections,
                              instrument=instrument, visit=visit, detector=ccd)
        except (LookupError):  # silently skip any ccds that don't exist
            for visit in visits[1:]:  # try the other visits just in case
                visit = int(visit)
                try:
                    bbox = butler.get('calexp.bbox', collections=collections,
                                      instrument=instrument, visit=visit,
                                      detector=ccd)
                except (LookupError):
                    continue
                break
        else:
            if instrument == 'DECam':
                camera = DarkEnergyCamera().getCamera()
            else:
                raise NotImplementedError
            cornerList.append([visit, ccd] + [val for pair in bbox.getCorners()
                              for val in ccd2focalPlane(pair[0], pair[1], ccd, camera)])
    corners = pd.DataFrame(cornerList)
    corners['width'] = corners[6] - corners[8]
    corners['height'] = corners[7] - corners[5]
    return corners


# TODO: DM-43095, to plotUtils
@deprecated(reason="This function is deprecated and will be removed soon.",
            version="v27", category=FutureWarning)
def getCcdAndVisitSizeOnSky(repo, sourceTable, instrument='DECam',
                            collections=[], visit=None, detector=None):
    """Estimate the area of one CCD and one visit on the sky in square degrees.

    Parameters
    ----------
    repo : `str`
        Repository corresponding to the output of an ap_pipe run.
    sourceTable : `pandas.core.frame.DataFrame`
        Pandas dataframe with DIA Sources from an APDB.
    instrument : `str`
        Default is 'DECam'
    collections : `list` or `str`
        Must be provided to load the camera properly
    visit : `int` or None, optional
        Specific visit to use when loading representative calexp.
    detector : `int` or None, optional
        Specific detector (ccd) to use when loading representative calexp.

    Returns
    -------
    ccdArea : `float`
        Area covered by one detector (CCD) on the sky, in square degrees
    visitArea :
        Area covered by a visit with all detectors (CCDs)
        on the sky, in square degrees
    """
    visits = np.unique(sourceTable.visit)
    ccds = np.unique(sourceTable.detector)
    nGoodCcds = len(ccds)
    butler = dafButler.Butler(repo)
    if visit is None:
        visit = int(visits[0])
    if detector is None:
        detector = int(ccds[0])
    calexp = butler.get('calexp', collections=collections,
                        instrument=instrument, visit=visit, detector=detector)
    bbox = butler.get('calexp.bbox', collections=collections,
                      instrument=instrument, visit=visit, detector=detector)
    pixelScale = calexp.getWcs().getPixelScale().asArcseconds()
    ccdArea = (pixelScale*pixelScale*bbox.getArea()*u.arcsec**2).to(u.deg**2).value
    visitArea = ccdArea * nGoodCcds
    return ccdArea, visitArea


# TODO: DM-43095, use an instantiated butler, and add to an appropriate
# visit-level (nightly?) AP pipeline.
@deprecated(reason="This function is deprecated and will be removed soon.",
            version="v27", category=FutureWarning)
def plotDiaSourceDensityInFocalPlane(repo, sourceTable, cmap=mpl.cm.Blues, title='',
                                     instrument='DECam', collections=[]):
    """Plot average density of DIA Sources in the focal plane (per CCD).

    Parameters
    ----------
    repo : `str`
        Repository corresponding to the output of an ap_pipe run.
    sourceTable : `pandas.core.frame.DataFrame`
        Pandas dataframe with DIA Sources from an APDB.
    cmap : `matplotlib.colors.ListedColormap`
        Matplotlib colormap.
    title : `str`
        String to append to the plot title, optional.
    instrument : `str`
        Default is 'DECam'
    collections : `list` or `str`
        Must be provided to load the camera properly
    """
    ccdArea, visitArea = getCcdAndVisitSizeOnSky(repo, sourceTable, instrument, collections)
    nVisits = len(np.unique(sourceTable['visit'].values))
    ccdGroup = sourceTable.groupby('detector')
    ccdSourceCount = ccdGroup.visit.count().values/nVisits/ccdArea
    # DIA Source count per visit per square degree, for each CCD
    butler = dafButler.Butler(repo)
    corners = getCcdCorners(butler, sourceTable, instrument, collections)
    norm = mpl.colors.Normalize(vmin=np.min(ccdSourceCount), vmax=np.max(ccdSourceCount))

    fig1 = plt.figure(figsize=(8, 8))
    ax1 = fig1.add_subplot(111, aspect='equal')

    for index, row in corners.iterrows():
        try:
            averageFocalPlane = ccdGroup.get_group(int(row[1])).x.count()/nVisits/ccdArea
        except KeyError:
            averageFocalPlane = 0  # plot normalization will be weird but it won't fall over
        ax1.add_patch(patches.Rectangle((row[7], row[6]),
                      -row.height,
                      -row.width,
                      fill=True,
                      color=cmap(norm(averageFocalPlane))))
        ax1.text(row[7]-row.height/2, row[6]-row.width/2, '%d' % (row[1]), fontsize=12)
        plt.plot(row[7]-row.height/2, row[6]-row.width/2, ',')
    ax1.set_title('Mean DIA Source density in focal plane coordinates %s' % (title))
    ax1.set_xlabel('Focal Plane X', size=16)
    ax1.set_ylabel('Focal Plane Y', size=16)
    ax1 = plt.gca()
    ax1.invert_yaxis()
    ax1.invert_xaxis()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, fraction=0.04, pad=0.04)
    cb.set_label('DIA Sources per sq. deg.', rotation=90)


# TODO: DM-43095, note this plots diaObjects and is different from HitsDiaPlot,
# yet it is HiTS-specific. For consistency, an analysis_tools style plotter
# probably ought to live in analysis_ap, not analysis_tools.
@deprecated(reason="This function is deprecated and will be removed soon.",
            version="v27", category=FutureWarning)
def plotHitsDiaObjectsOnSky(title, objects, row_mask):
    fig = plt.figure(facecolor='white', figsize=(10, 8))

    dec_set1 = (objects['decl'] > -2) & row_mask
    dec_set2 = (~dec_set1) & row_mask
    plt.subplots_adjust(wspace=0.1, hspace=0)

    # Panel 1: one HiTS field, on the right
    ax1 = plt.subplot2grid((100, 100), (0, 55), rowspan=90, colspan=45)
    plot1 = ax1.scatter(objects.loc[dec_set1, 'ra'], objects.loc[dec_set1, 'decl'],
                        marker='.', lw=0, s=objects.loc[dec_set1, 'nDiaSources']*8,
                        c=objects.loc[dec_set1, 'nDiaSources'], alpha=0.7,
                        cmap='viridis', linewidth=0.5, edgecolor='k')
    plt.xlabel('RA (deg)', size=16)
    plt.ylabel('Dec (deg)', size=16)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position('right')
    ax1.invert_xaxis()  # RA should increase to the left
    loc = plticker.MultipleLocator(base=0.5)  # puts ticks at regular intervals
    ax1.yaxis.set_major_locator(loc)
    cb = fig.colorbar(plot1, orientation='horizontal', pad=0.3)
    cb.set_label('Number of Sources per Object', size=16)
    cb.set_clim(np.min(objects.loc[dec_set2, 'nDiaSources']),
                np.max(objects.loc[dec_set2, 'nDiaSources']))
    cb.solids.set_edgecolor("face")
    cb.remove()  # don't show colorbar by this panel

    # Panel 2: two (overlapping) HiTS fields, on the left
    ax2 = plt.subplot2grid((100, 100), (0, 0), rowspan=90, colspan=50)
    plot2 = ax2.scatter(objects.loc[dec_set2, 'ra'], objects.loc[dec_set2, 'decl'],
                        marker='.', lw=0, s=objects.loc[dec_set2, 'nDiaSources']*8,
                        c=objects.loc[dec_set2, 'nDiaSources'], alpha=0.7,
                        cmap='viridis', linewidth=0.5, edgecolor='k')
    plt.xlabel('RA (deg)', size=16)
    plt.ylabel('Dec (deg)', size=16)
    plt.title(title, size=16)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.invert_xaxis()
    cax = plt.subplot2grid((100, 100), (60, 55), rowspan=5, colspan=45)
    cb2 = fig.colorbar(plot2, cax=cax, orientation='horizontal', pad=0.1)
    cb2.set_label('Number of Sources per Object', size=16)
    cb2.set_clim(np.min(objects.loc[dec_set2, 'nDiaSources']),
                 np.max(objects.loc[dec_set2, 'nDiaSources']))
    cb2.solids.set_edgecolor("face")


# TODO: as part of DM-43201, make sure the sole analysis_ap plotting
# pipeline works, then delete this.
@deprecated(reason="This function is deprecated and will be removed soon.",
            version="v27", category=FutureWarning)
def plotHitsSourcesOnSky(sourceTable, title=''):
    """Plot DIA Sources from three DECam HiTS fields on the sky.

    Can also be used to plot DIA Objects instead of DIA Sources.

    Parameters
    ----------
    sourceTable : `pandas.core.frame.DataFrame`
        Pandas dataframe with DIA Sources from an APDB.
    title : `str`
        Title for the plot, optional.
    """
    plt.figure(figsize=(9, 7))
    ax1 = plt.subplot2grid((100, 100), (0, 55),
                           rowspan=50, colspan=45)  # 1 single HiTS field, on the right
    ax2 = plt.subplot2grid((100, 100), (0, 0),
                           rowspan=90, colspan=50)  # 2 overlapping HiTS fields, on the left

    ax1Filter = (sourceTable['dec'] > -2)
    ax2Filter = (~ax1Filter)

    ax1.scatter(sourceTable.loc[ax1Filter, 'ra'],
                sourceTable.loc[ax1Filter, 'dec'],
                c='C0', s=2, alpha=0.2)
    ax2.scatter(sourceTable.loc[ax2Filter, 'ra'],
                sourceTable.loc[ax2Filter, 'dec'],
                c='C0', s=2, alpha=0.2)

    ax1.set_xlabel('RA (deg)')
    ax2.set_xlabel('RA (deg)')
    ax1.set_ylabel('Dec (deg)')
    ax2.set_ylabel('Dec (deg)')

    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax1.set_xlim(151.2, 148.9)

    ax2.set_xlim(156.6, 153.9)
    ax2.set_ylim(-7.6, -3.9)

    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position('right')

    plt.title(title)
    plt.subplots_adjust(wspace=0.1, hspace=0)


# TODO: DM-43095, use analysis_tools' existing HistPlot, no hard-wired values,
# decide how to best choose a representative seeing value for a whole visit
@deprecated(reason="This function is deprecated and will be removed soon.",
            version="v27", category=FutureWarning)
def plotSeeingHistogram(repo, sourceTable, ccd=35, instrument='DECam', collections=[]):
    """Plot distribution of visit seeing.

    Parameters
    ----------
    repo : `str`
        Repository corresponding to the output of an ap_pipe run.
    sourceTable : `pandas.core.frame.DataFrame`
        Pandas dataframe with DIA Sources from an APDB.
    ccd : `int`
        The ccd being considered, default 35.
    instrument : `str`, optional
        Default is 'DECam'
    collections : `list` or `str`
        Must be provided to load the camera properly
    """
    fwhm = pd.DataFrame()
    visits = np.unique(sourceTable['visit'])
    radii = []

    butler = dafButler.Butler(repo)
    for visit in visits:
        psf = butler.get('calexp.psf', instrument=instrument,
                         collections=collections,
                         visit=int(visit), detector=ccd)
        psfSize = psf.computeShape().getDeterminantRadius()
        radii.append(psfSize*2.355)  # convert sigma to FWHM
        # Get just one calexp for WCS purposes
    calexp = butler.get('calexp', collections=collections,
                        instrument=instrument,
                        visit=int(visit), detector=ccd)
    fwhm['visit'] = pd.Series(visits)
    fwhm['radius'] = pd.Series(radii, index=fwhm.index)
    pixelScale = calexp.getWcs().getPixelScale().asArcseconds()  # same for all visits
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.hist(fwhm['radius'].values, alpha=0.5)
    plt.xlabel('Seeing FWHM (pixels)')
    plt.ylabel('Visit count')
    secax = ax.secondary_xaxis('top', functions=(lambda x: x*pixelScale, lambda x: x/pixelScale))
    secax.set_xlabel('Seeing FWHM (arcseconds)')


# TODO: DM-43095, consider adding to relevant AP pipeline
@deprecated(reason="This function is deprecated and will be removed soon.",
            version="v27", category=FutureWarning)
def plotDiaSourcesInFocalPlane(repo, sourceTable, gridsize=(400, 400), title='',
                               instrument='DECam', collections=[]):
    """Plot DIA Source locations in the focal plane.

    Parameters
    ----------
    repo : `str`
        Repository corresponding to the output of an ap_pipe run.
    sourceTable : `pandas.core.frame.DataFrame`
        Pandas dataframe with DIA Sources from an APDB.
    gridsize : `tuple` of form (int, int)
        Number of hexagons in the (x, y) directions for the hexbin plot.
    title : `str`
        String to append to the plot title, optional.
    instrument : `str`
        Default is 'DECam'
    collections : `list` or `str`
        Must be provided to load the camera properly
    """
    butler = dafButler.Butler(repo)
    if instrument == 'DECam':
        camera = DarkEnergyCamera().getCamera()
    else:
        raise NotImplementedError
    corners = getCcdCorners(butler, sourceTable, instrument, collections)
    xFP_list = []
    yFP_list = []
    for index, row in sourceTable.iterrows():
        xFP, yFP = ccd2focalPlane(row['x'], row['y'], row['detector'], camera=camera)
        xFP_list.append(xFP)
        yFP_list.append(yFP)
    xFP_Series = pd.Series(xFP_list, index=sourceTable.index)
    yFP_Series = pd.Series(yFP_list, index=sourceTable.index)

    fig1 = plt.figure(figsize=(8, 8))
    ax1 = fig1.add_subplot(111, aspect='equal')

    for index, row in corners.iterrows():
        ax1.add_patch(patches.Rectangle((row[7], row[6]),
                                        -row.height,
                                        -row.width,
                                        fill=False))
        ax1.text(row[7] - row.height/2, row[6] - row.width/2, '%d' % (row[1]))
        plt.plot(row[7] - row.height/2, row[6] - row.width/2, ',')

    # somehow x and y are switched... geometry is hard
    ax1.hexbin(yFP_Series, xFP_Series, gridsize=gridsize, bins='log', cmap='Blues')
    ax1.set_title('DIA Sources in focal plane coordinates %s' % (title))

    ax1.set_xlabel('Focal Plane X', size=16)
    ax1.set_ylabel('Focal Plane Y', size=16)
    ax1.invert_yaxis()
    ax1.invert_xaxis()


# TODO: DM-43095, multiple calls to DiaSkyPlot
# may be memory-intensive for large numbers of visits so be careful
@deprecated(reason="This function is deprecated and will be removed soon.",
            version="v27", category=FutureWarning)
def plotDiaSourcesOnSkyGrid(sourceTable, title=None, color='C0', size=0.1):
    """Make a multi-panel plot of DIA Sources for each visit on the sky.

    Parameters
    ----------
    sourceTable : `pandas.core.frame.DataFrame`
        Pandas dataframe with DIA Sources from an APDB.
    title : `str`
        String for overall figure title, optional.
    color : `str`
        Color to use for the scatter plot, optional (default is C0 blue).
    size : `float`
        Size for points with marker style '.'.
    """
    visits = np.unique(sourceTable['visit'])
    nVisits = len(visits)
    if np.floor(np.sqrt(nVisits)) - np.sqrt(nVisits) == 0:
        squareGridSize = np.int(np.sqrt(nVisits))
    else:
        squareGridSize = np.int(np.sqrt(nVisits)) + 1
    fig = plt.figure(figsize=(9, 9))
    for count, visit in enumerate(np.unique(sourceTable['visit'].values)):
        idx = sourceTable.visit == visit
        ax = fig.add_subplot(squareGridSize, squareGridSize, count + 1, aspect='equal')
        ax.scatter(sourceTable.ra[idx], sourceTable.dec[idx], c=color,
                   marker='.', s=size, alpha=0.2)
        ax.set_title(visit, size=8)
        ax.invert_xaxis()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(wspace=0)
    if title:
        fig.suptitle(title)


# TODO: DM-43095, and add to an appropriate AP pipeline
@deprecated(reason="This function is deprecated and will be removed soon.",
            version="v27", category=FutureWarning)
def plotFlagHist(sourceTable, title=None,
                 badFlagList=['base_PixelFlags_flag_bad',
                              'base_PixelFlags_flag_suspect',
                              'base_PixelFlags_flag_saturatedCenter']):
    """Plot a histogram of how often each pixel flag occurs in DIA Sources.

    Parameters
    ----------
    sourceTable : `pandas.core.frame.DataFrame`
        Pandas dataframe with DIA Sources from an APDB.
    title : `str`
        String for overall figure title, optional.
    badFlagList : `list`, optional
        Flag names to plot in red, presumed to indicate a garbage DIA Source.
    """
    config = TransformDiaSourceCatalogConfig()
    unpacker = UnpackApdbFlags(config.flagMap, 'DiaSource')
    flagValues = unpacker.unpack(sourceTable['flags'], 'flags')
    labels = list(flagValues.dtype.names)
    flagTable = pd.DataFrame(flagValues, index=sourceTable.index)
    flagSum = flagTable.sum()
    flagsToPlot = [count for count in flagSum.values]
    assert len(flagsToPlot) == len(labels)

    flagColors = []
    for label in labels:
        if label in badFlagList:
            flagColors.append('C3')
        else:
            flagColors.append('C0')

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.barh(labels, flagsToPlot, color=flagColors)
    fig.subplots_adjust(left=0.35)
    ax.set_xlabel('Number of flagged DIASources')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title(title)


# TODO: DM-42824
@deprecated(reason="This function is deprecated and will be removed soon.",
            version="v27", category=FutureWarning)
def plotFluxHistSrc(srcTable1, srcTable2=None, fluxType='psFlux',
                    label1=None, label2=None, title=None, ylog=False,
                    color1='#2979C1', color2='#Bee7F5',
                    binmin=-3e3, binmax=3e3, nbins=200):
    """Plot distribution of fluxes from 1-2 DIASource tables.

    Parameters
    ----------
    srcTable1 : `pandas.core.frame.DataFrame`
        Pandas dataframe with DIA Sources from an APDB.
    srcTable2 : `pandas.core.frame.DataFrame`, optional
        Second pandas dataframe with DIA Sources from an APDB.
    fluxType : `str`, optional
        Choose from psFlux (default), totFlux, or apFlux.
    label1 : `str`, optional
        Label for srcTable1.
    label2 : `str`, optional
        Label for srcTable2.
    title : `str`, optional
        Plot title.
    ylog : `bool`, optional
        Plot the y-axis on a log scale? Default False
    color1 : `str`, optional
        Color for srcTable1.
    color2 : `str`, optional
        Color for srcTable2.
    binmin : `float`, optional
        Minimum x-value for bins.
    binmax : `float`, optional
        Maximum x-value for bins.
    nbins : `int`, optional
        Number of histogram bins.

    """
    plt.figure()
    plt.xlabel(fluxType, size=12)
    plt.ylabel('DIA Source count', size=12)
    bins = np.linspace(binmin, binmax, nbins)
    if ylog:
        plt.yscale('log')
    plt.hist(srcTable1[fluxType].values, bins=bins, color=color1, label=label1)
    if srcTable2 is not None:
        plt.hist(srcTable2[fluxType].values, bins=bins, color=color2, label=label2)
    if label1:
        plt.legend(frameon=False, fontsize=12)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title(title)


# TODO: DM-42824
@deprecated(reason="This function is deprecated and will be removed soon.",
            version="v27", category=FutureWarning)
def source_magnitude_histogram(repo, sourceTable, bandToPlot, instrument,
                               collections, detectorToUse=42, binsize=0.2,
                               min_magnitude=None, max_magnitude=None,
                               badFlagList=['base_PixelFlags_flag_bad',
                                            'base_PixelFlags_flag_suspect',
                                            'base_PixelFlags_flag_saturatedCenter']):
    """Plot magnitudes of sources from the source catalog for DIA Sources.

    Note that the values plotted are from the 'src' catalog corresponding to
    the processed visit image, not the APDB or difference image source catalog.

    Parameters
    ----------
    repo : `str`
        Gen3 repository containing 'collections'
    sourceTable : `pandas.core.frame.DataFrame`
        Pandas dataframe with DIA Sources from an APDB.
    bandToPlot : `str`
        Typically one of 'g', 'r', 'i', 'z', or 'y'
    instrument : `str`
        e.g., 'HSC' or 'DECam'
    collections : `str`
        Collection within gen3 'repo' to use
    detectorToUse : `int`, optional
        Detector to use for all the plots, default 42
    binsize : `float`, optional
        Histogram bin size, default 0.2
    min_magnitude : `float` or None, optional
        Set plot x-axis minimum
    max_magnitude : `float` or None, optional
        Set plot x-axis maximum
    badFlagList : `list`, optional
        Exclude sources with flags in this list.
    """

    visits = np.unique(sourceTable['visit'])
    area, _ = getCcdAndVisitSizeOnSky(repo, sourceTable, detector=detectorToUse,
                                      instrument=instrument, collections=collections)
    butler = dafButler.Butler(repo)

    fig, ax = plt.subplots(figsize=(8, 5))
    for visit in visits:
        band = sourceTable.loc[sourceTable['visit'] == visit, 'filterName'].values[0]
        if band != bandToPlot:
            continue
        if band == 'g':
            color = 'C2'
        elif band == 'r':
            color = 'C1'
        elif band == 'i':
            color = 'C3'
        elif band == 'z':
            color = 'C5'
        else:
            color = 'k'  # should be y
        src = butler.get("src", dataId={"visit": visit, "detector": detectorToUse},
                         collections=collections, instrument=instrument)
        flag_src = [False]*len(src)
        f_nJy = src['base_PsfFlux_instFlux']
        for flag in badFlagList:
            flag_src |= src[flag]

        good_fluxes = np.array([s for s, f in zip(f_nJy, flag_src) if ~f and s > 0])
        mags = (good_fluxes*u.nJy).to_value(u.ABmag)

        if min_magnitude is None:
            min_magnitude = np.floor(np.min(mags)/binsize)*binsize
        if max_magnitude is None:
            max_magnitude = np.ceil(np.max(mags)/binsize)*binsize
        nbins = int((max_magnitude - min_magnitude)/binsize)
        hist, bin_edges = np.histogram(mags, bins=nbins, range=(min_magnitude, max_magnitude))
        bin_centers = [(bin_edges[i] + bin_edges[i + 1])/2 for i in range(len(bin_edges) - 1)]
        plt.plot(bin_centers, hist/area/binsize, label=visit, color=color)
    plt.title(f'{bandToPlot} Source Counts')
    plt.xlabel('Magnitude')
    plt.ylabel('Detected sources per mag per deg^2')
    ax.set_yscale('log')


# TODO: DM-40203
@deprecated(reason="This function is deprecated and will be removed soon.",
            version="v27", category=FutureWarning)
def plotTractOutline(tractInfo, patchList, axes=None, fontSize=10,
                     maxDegBeyondPatch=1.5):
    """Plot the the outline of the tract and patches highlighting
    those with data.
    As some skyMap settings can define tracts with a large number of patches,
    this can become very crowded.
    So, if only a subset of patches are included, find the outer boundary
    of all patches in patchList and only plot to maxDegBeyondPatch degrees
    beyond those boundaries (in all four directions).

    Parameters
    ----------
    tractInfo : `lsst.skymap.tractInfo.ExplicitTractInfo`
       Tract information object for extracting tract RA and DEC limits.
    patchList : `list` of `str`
       List of patch IDs with data to be plotted.  These will be color
       shaded in the outline plot.
    fontSize : `int`
       Font size for plot labels.
    maxDegBeyondPatch : `float`
       Maximum number of degrees to plot beyond the border defined by all
       patches with data to be plotted.
    """
    def percent(values, p=0.5):
        """Return a value a fraction of the way between the min and max
        values in a list.
        """
        interval = max(values) - min(values)
        return min(values) + p*interval

    if axes is None:
        fig = plt.figure(figsize=(8, 8))
        axes = fig.gca()
    buff = 0.02
    axes.tick_params(which="both", direction="in", labelsize=fontSize)
    axes.locator_params(nbins=6)
    axes.ticklabel_format(useOffset=False)

    tractRa, tractDec = bboxToRaDec(tractInfo.getBBox(), tractInfo.getWcs())
    patchBoundary = getRaDecMinMaxPatchList(patchList, tractInfo,
                                            pad=maxDegBeyondPatch)

    xMin = min(max(tractRa), patchBoundary.raMax) + buff
    xMax = max(min(tractRa), patchBoundary.raMin) - buff
    yMin = max(min(tractDec), patchBoundary.decMin) - buff
    yMax = min(max(tractDec), patchBoundary.decMax) + buff
    xlim = xMin, xMax
    ylim = yMin, yMax
    axes.fill(tractRa, tractDec, fill=True, color="black", lw=1, linestyle='solid',
              edgecolor='k', alpha=0.2)
    for ip, patch in enumerate(tractInfo):
        patchIndexStr = str(patch.getIndex()) + ' ' + str(tractInfo.getSequentialPatchIndex(patch))
        patchWithData = tractInfo.getSequentialPatchIndex(patch)
        color = "k"
        alpha = 0.05
        if patchWithData in patchList:
            # color = ("c", "g", "r", "b", "m")[ip%5]
            color = 'g'
            alpha = 0.5
        ra, dec = bboxToRaDec(patch.getOuterBBox(), tractInfo.getWcs())
        deltaRa = abs(max(ra) - min(ra))
        deltaDec = abs(max(dec) - min(dec))
        pBuff = 0.5*max(deltaRa, deltaDec)
        centerRa = min(ra) + 0.5*deltaRa
        centerDec = min(dec) + 0.5*deltaDec
        if (centerRa < xMin + pBuff and centerRa > xMax - pBuff
                and centerDec > yMin - pBuff and centerDec < yMax + pBuff):
            axes.fill(ra, dec, fill=True, color=color, lw=1,
                      linestyle="solid", alpha=alpha)
            if patchWithData in patchList or (centerRa < xMin - 0.2*pBuff
                                              and centerRa > xMax + 0.2*pBuff
                                              and centerDec > yMin + 0.2*pBuff
                                              and centerDec < yMax - 0.2*pBuff):
                axes.text(percent(ra), percent(dec, 0.5), str(patchIndexStr),
                          fontsize=fontSize - 1, horizontalalignment="center",
                          verticalalignment="center")
    axes.set_xlabel('RA', size='large')
    axes.set_ylabel('Dec', size='large')
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)


# TODO: DM-40203
@deprecated(reason="This function is deprecated and will be removed soon.",
            version="v27", category=FutureWarning)
def mosaicCoadd(repo, patch_list, band='g', tractIndex=0, refPatchIndex=None,
                sampling=100, norm=None, nImage=False, fig=None,
                show_colorbar=True, filename=None, flipX=True, flipY=False,
                instrument=None, collections=None, skymapName=None,
                coaddName='deepCoadd'):
    """Generate a mosaic image of many coadded patches. Gen3 only.

    Parameters
    ----------
    repo : `str`
        The path to the data repository.
    patch_list : `list`
        A list of the patch indices containing images to mosaic.
        List elements will be `float` (sequential patch indices).
    band : `str`, optional
        The band of the coadd to retrieve from the repository.
    tractIndex : `int`, optional
        The tract of the skyMap.
    refPatchIndex : `str`, optional
        If set, use the given patch to set the image normalization
        for the figure.
    sampling : `int`, optional
        Stride factor to sample each input image in order to reduce the
        size in memory.
        A `sampling` of 1 will attempt to display every pixel.
    norm : `astropy.visualization.ImageNormalize`, optional
        Normalization to set the color scale of the images.
        If `None`, the normalization will be calculated from the first image.
        If you wish to use any normalization other than zscale, you must
        calculate it ahead of time and pass it in as `norm` here.
    nImage : `bool`, optional
        Mosaic the nImage instead of the coadd.
    fig : `matplotlib.pyplot.fig`, optional
        Figure instance to display the mosaic in.
    show_colorbar : `bool`, optional
        Display a colorbar on the figure.
    filename : `str`, optional
        If set, write the figure to a file with the given filename.
    flipX : `bool`, optional
        Set to flip the individual patch images horizontally.
    flipY : `bool`, optional
        Set to flip the individual patch images vertically.
    instrument : `str`, optional
        Default is 'DECam'.
    collections : `list` or `str`, optional
        Must be provided to load the camera properly
    skymapName : `str`, optional
        Must be provided to load the skymap. e.g., 'hsc_rings_v1'
    coaddName : `str`, optional
        Type of coadd, default `deepCoadd` (you might want `goodSeeingCoadd`)
    """
    # Decide what data product to plot (actual pixel data or nImages)
    if nImage:
        coaddName += '_nImage'

    # Create a mapping between sequential patch indices and (x,y) patch indices
    if not collections:
        raise ValueError('One or more collections is required.')
    if not skymapName:
        raise ValueError('A skymap is required.')
    if not instrument:
        raise ValueError('An instrument is required.')
    butler = dafButler.Butler(repo, collections=collections)
    skymap = butler.get('skyMap', skymap=skymapName, instrument=instrument,
                        collections='skymaps')
    tract = skymap.generateTract(tractIndex)
    patchesToLoad = [patch for patch in tract if tract.getSequentialPatchIndex(patch) in patch_list]
    patchIndicesToLoad = [patch.getIndex() for patch in patchesToLoad]  # (x,y) indices
    indexmap = {}
    for patch in tract:
        indexmap[str(tract.getSequentialPatchIndex(patch))] = patch.getIndex()

    # Use a reference patch to set the normalization for all the patches
    if norm is None:
        if refPatchIndex in patchIndicesToLoad:
            sequentialPatchIndex = [key for key, value in indexmap.items() if value == refPatchIndex][0]
            dataId = {'band': band, 'tract': tractIndex,
                      'patch': int(sequentialPatchIndex),
                      'instrument': instrument, 'skymap': skymapName}
            if nImage:
                coaddArray = butler.get(coaddName, dataId=dataId).getArray()
            else:
                coaddArray = butler.get(coaddName, dataId=dataId).getImage().getArray()
            norm = ImageNormalize(coaddArray, interval=ZScaleInterval(),
                                  stretch=SqrtStretch())

    # Set up the figure grid
    patch_x = []
    patch_y = []
    for patch in patchesToLoad:
        patch_x.append(patch.getIndex()[0])
        patch_y.append(patch.getIndex()[1])
    x0 = min(patch_x)
    y0 = min(patch_y)
    nx = max(patch_x) - x0 + 1
    ny = max(patch_y) - y0 + 1
    fig = plt.figure(figsize=(nx, ny), constrained_layout=False)
    gs1 = fig.add_gridspec(ny, nx, wspace=0, hspace=0)

    # Plot coadd patches with data and print the patch index if there's no data
    for x in range(nx):
        for y in range(ny):
            figIdx = x + nx*y
            ax = fig.add_subplot(gs1[figIdx])
            ax.set_aspect('equal', 'box')
            patchIndex = (x, ny - y - 1)
            if patchIndex in patchIndicesToLoad:
                sequentialPatchIndex = [key for key, value in indexmap.items() if value == patchIndex][0]
                dataId = {'band': band, 'tract': tractIndex, 'patch': int(sequentialPatchIndex),
                          'instrument': instrument, 'skymap': skymapName}
                try:
                    coadd = butler.get(coaddName, dataId=dataId)
                except LookupError:
                    print(f'Failed to retrieve data for patch {patchIndex}')
                    ax.text(.3, .3, patchIndex)
                    continue
                if nImage:
                    coaddArray = coadd.getArray()
                else:
                    coaddArray = coadd.getImage().getArray()
                coaddArray = coaddArray[::sampling, ::sampling]
                if flipX:
                    coaddArray = np.flip(coaddArray, axis=0)
                if flipY:
                    coaddArray = np.flip(coaddArray, axis=1)
                if norm is None:
                    norm = ImageNormalize(coaddArray,
                                          interval=ZScaleInterval(),
                                          stretch=SqrtStretch())
                im = ax.imshow(coaddArray, cmap='gray', norm=norm)
            else:
                ax.text(.3, .3, patchIndex)
                im = None
                # print('No data in patch', patchIndex)
            plt.setp(ax, xticks=[], yticks=[])

    # Adjust and annotate plot
    if show_colorbar and (im is not None):
        cbar_width = 0.01
        cbar_height = 0.5
        cbar_ax = fig.add_axes([0.9 - cbar_width, 0.5 - cbar_height/2,
                                cbar_width, cbar_height])
        fig.colorbar(im, cax=cbar_ax)

    # Save figure, if desired
    if filename:
        try:
            plt.savefig(filename, transparent=True)
        except Exception as e:
            print(f"Could not write file '{filename}': {e}")

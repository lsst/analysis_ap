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
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.patches as patches
import pandas as pd

from astropy import units as u
from astropy import visualization as aviz

import lsst.daf.butler as dafButler
import lsst.geom
from lsst.ap.association import UnpackApdbFlags, TransformDiaSourceCatalogConfig
import lsst.afw.cameraGeom as cameraGeom
from lsst.obs.decam import DarkEnergyCamera

from .legacyApdbUtils import loadExposures


def plotDiaObjectHistogram(objTable, objFiltered,
                           label1='All Objects',
                           label2='Filtered Objects',
                           title=''):
    """Create a histogram showing how many DIA Sources
    comprise the DIA Objects.

    Parameters
    ----------
    objTable : `pandas.DataFrame`
        DIA Object Table.
    objFiltered : `pandas.core.frame.DataFrame`
        DIA Object Table that is a filtered subset of objTable.
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
    binMax = np.max(objTable['nDiaSources'].values)
    plt.hist(objTable['nDiaSources'].values, bins=np.arange(0, binMax),
             color='#2979C1', label=label1)
    plt.hist(objFiltered['nDiaSources'].values, bins=np.arange(0, binMax),
             color='#Bee7F5', label=label2)
    plt.legend(frameon=False, fontsize=16)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title(title)
    return fig


def plotDiaObjectsOnSky(rerunName, objTable, objFilter, hits):
    """Create a plot of filtered DIAObjects on the sky.

    Parameters
    ----------
    rerunName : `str`
        Name of the directory at the end of the repo path.
    objTable : `pandas.DataFrame`
        DIA Object Table.
    objFilter : `pandas.Series` of `bool`
        Filter applied to create a subset (e.g., quality cut) from objTable.
    hits : `boolean`
        True for two panels with custom axes for the hits2015 dataset
        False for a single plot with automatic axis limits

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Scatter plot of the DIAObjects on the sky with RA, Dec axes.
        Point sizes and colors correspond to the number of sources per object.
    """
    fig = plt.figure(facecolor='white', figsize=(10, 8))

    if hits:  # two subplots
        dec_set1 = (objTable['decl'] > -2) & objFilter
        dec_set2 = (~dec_set1) & objFilter
        plt.subplots_adjust(wspace=0.1, hspace=0)

        # Panel 1: one HiTS field, on the right
        ax1 = plt.subplot2grid((100, 100), (0, 55), rowspan=90, colspan=45)
        plot1 = ax1.scatter(objTable.loc[dec_set1, 'ra'], objTable.loc[dec_set1, 'decl'],
                            marker='.', lw=0, s=objTable.loc[dec_set1, 'nDiaSources']*8,
                            c=objTable.loc[dec_set1, 'nDiaSources'], alpha=0.7,
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
        cb.set_clim(np.min(objTable.loc[dec_set2, 'nDiaSources']),
                    np.max(objTable.loc[dec_set2, 'nDiaSources']))
        cb.solids.set_edgecolor("face")
        cb.remove()  # don't show colorbar by this panel

        # Panel 2: two (overlapping) HiTS fields, on the left
        ax2 = plt.subplot2grid((100, 100), (0, 0), rowspan=90, colspan=50)
        plot2 = ax2.scatter(objTable.loc[dec_set2, 'ra'], objTable.loc[dec_set2, 'decl'],
                            marker='.', lw=0, s=objTable.loc[dec_set2, 'nDiaSources']*8,
                            c=objTable.loc[dec_set2, 'nDiaSources'], alpha=0.7,
                            cmap='viridis', linewidth=0.5, edgecolor='k')
        plt.xlabel('RA (deg)', size=16)
        plt.ylabel('Dec (deg)', size=16)
        plt.title(rerunName, size=16)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.invert_xaxis()
        cax = plt.subplot2grid((100, 100), (60, 55), rowspan=5, colspan=45)
        cb2 = fig.colorbar(plot2, cax=cax, orientation='horizontal', pad=0.1)
        cb2.set_label('Number of Sources per Object', size=16)
        cb2.set_clim(np.min(objTable.loc[dec_set2, 'nDiaSources']),
                     np.max(objTable.loc[dec_set2, 'nDiaSources']))
        cb2.solids.set_edgecolor("face")

    else:  # one main plot
        ax = fig.add_subplot(111)
        plot = ax.scatter(objTable.loc[objFilter, 'ra'],
                          objTable.loc[objFilter, 'decl'], marker='.', lw=0,
                          s=objTable.loc[objFilter, 'nDiaSources']*8,
                          c=objTable.loc[objFilter, 'nDiaSources'],
                          alpha=0.7, cmap='viridis', linewidth=0.5, edgecolor='k')
        plt.xlabel('RA (deg)', size=16)
        plt.ylabel('Dec (deg)', size=16)
        plt.title(rerunName, size=16)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.invert_xaxis()  # RA should increase to the left
        cb = fig.colorbar(plot, orientation='horizontal')
        cb.set_label('Number of Sources per Object', size=16)
        cb.set_clim(np.min(objTable.loc[objFilter, 'nDiaSources']),
                    np.max(objTable.loc[objFilter, 'nDiaSources']))
        cb.solids.set_edgecolor("face")
    return fig


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

    ax1Filter = (sourceTable['decl'] > -2)
    ax2Filter = (~ax1Filter)

    ax1.scatter(sourceTable.loc[ax1Filter, 'ra'],
                sourceTable.loc[ax1Filter, 'decl'],
                c='C0', s=2, alpha=0.2)
    ax2.scatter(sourceTable.loc[ax2Filter, 'ra'],
                sourceTable.loc[ax2Filter, 'decl'],
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


def plotDiaSourcesOnSkyGrid(repo, sourceTable, title=None, color='C0', size=0.1):
    """Make a multi-panel plot of DIA Sources for each visit on the sky.

    Parameters
    ----------
    repo : `str`
        Repository corresponding to the output of an ap_pipe run.
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
        ax.scatter(sourceTable.ra[idx], sourceTable.decl[idx], c=color,
                   marker='.', s=size, alpha=0.2)
        ax.set_title(visit, size=8)
        ax.invert_xaxis()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(wspace=0)
    if title:
        fig.suptitle(title)


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


# Stuff from plotLightCurve.py

def retrieveCutouts(butler, dataId, collections, center, size=lsst.geom.Extent2I(30, 30), diffName='deep'):
    """Return small cutout exposures for a science exposure, difference image,
    and warped template.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler in the repository corresponding to the output of an ap_pipe run.
    dataId : `dict`-like
        Gen3 data ID specifying at least instrument, visit, and detector.
    collections : `str` or `list`
        Gen3 collection or collections from which to load the exposures.
    center : `lsst.geom.SpherePoint`
        Desired center coordinate of cutout.
    size : `lsst.geom.Extent`, optional
        Desired size of cutout, default is 30x30 pixels
    diffName : `str`, optional
        Default is 'deep', but 'goodSeeing' may be needed instead.

    Returns
    -------
    scienceCutout : `lsst.afw.Exposure`
        Cutout of calexp at location 'center' of size 'size'.
    differenceCutout : `lsst.afw.Exposure`
        Cutout of diffName_differenceExp at location 'center' of size 'size'.
    templateCutout : `lsst.afw.Exposure`
        Cutout of diffName_templateExp at location 'center' of size 'size'.
    """
    science, difference, template = loadExposures(butler, dataId,
                                                  collections, diffName)
    scienceCutout = science.getCutout(center, size)
    differenceCutout = difference.getCutout(center, size)
    templateCutout = template.getCutout(center, size)
    return scienceCutout, differenceCutout, templateCutout


def plotCutout(scienceCutout, differenceCutout, templateCutout, output=None):
    """Plot the cutouts for one DIASource in one image.

    Parameters
    ----------
    scienceCutout : `lsst.afw.Exposure`
        Cutout of calexp returned by retrieveCutouts.
    differenceCutout : `lsst.afw.Exposure`
        Cutout of deepDiff_differenceExp returned by retrieveCutouts.
    templateCutout : `lsst.afw.Exposure`
        Cutout of deepDiff_templateExp returned by retrieveCutouts.
    output : `str`, optional
        If provided, save png to disk at output filepath.
    """
    def do_one(ax, data, name):
        interval = aviz.ZScaleInterval()
        if name == 'Difference':
            norm = aviz.ImageNormalize(data, stretch=aviz.LinearStretch())
        else:
            norm = aviz.ImageNormalize(data, interval=interval, stretch=aviz.AsinhStretch(a=0.01))
        ax.imshow(data, cmap=cm.bone, interpolation="none", norm=norm)
        ax.axis('off')
        ax.set_title(name)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    do_one(ax1, templateCutout.image.array, "Template")
    do_one(ax2, scienceCutout.image.array, "Science")
    do_one(ax3, differenceCutout.image.array, "Difference")
    plt.tight_layout()

    if output is not None:
        plt.savefig(output, bbox_inches="tight")
        plt.close()


def setObjectFilter(objTable):
    """Define a subset of objects to plot, i.e., make some kind of quality cut.

    The definition of objFilter is presently hard-wired here.

    Parameters
    ----------
    objTable : `pandas.DataFrame`
        DIA Object Table.

    Returns
    -------
    objFilter : `pandas.Series` of `bool`
        Filter applied to create a subset (e.g., quality cut) from objTable.
    """
    objFilter = ((objTable['nDiaSources'] > 14) & (objTable['flags'] == 0))
    numTotal = len(objTable['diaObjectId'])
    numFilter = len(objTable.loc[objFilter, 'diaObjectId'])
    print('There are {0} total DIAObjects and {1} filtered DIAObjects.'.format(numTotal, numFilter))
    return objFilter

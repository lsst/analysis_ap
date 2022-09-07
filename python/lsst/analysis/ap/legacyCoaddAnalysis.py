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
import matplotlib.pyplot as plt
from astropy.visualization import (ZScaleInterval, SqrtStretch, ImageNormalize)

from lsst.pipe.base import Struct
import lsst.daf.butler as dafButler

__all__ = ("getPatchConstituents", "plotTractOutline", "mosaicCoadd")


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

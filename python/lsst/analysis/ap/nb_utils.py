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

from __future__ import annotations

__all__ = ["make_simbad_link", "compare_sources", "compare_objects",
           "find_objects_sharing_sources",
           "classify_association_clusters",
           "plot_cutouts_with_object_markers",
           "plot_objects_sharing_sources",
           "display_images", "display_images_ab",
           "get_xy_from_source_table", "extract_timestamped_messages"]

import astropy.coordinates as coord
from astroquery.simbad import Simbad
import astropy.units as u
import astropy.table
from datetime import datetime, timezone
import functools
import json
import numpy as np
import os
import pandas as pd
from typing import Any

import lsst.afw.display
from lsst.daf.butler import DatasetNotFoundError
from lsst.analysis.ap import plotImageSubtractionCutouts
from lsst.analysis.ap.compare import match_catalogs
from IPython.display import display, Image, Markdown


# Maps the image_type kwarg used by `display_images` to butler dataset
# names.
_IMAGE_DATASETS = {
    "science": "preliminary_visit_image",
    "template": "template_detector",
    "difference": "difference_image",
}


def _cutout_exists(cpath, dia_source_id):
    """Return True if a cutout PNG for this diaSourceId already exists.

    Parameters
    ----------
    cpath : `~plotImageSubtractionCutouts.CutoutPath`
        Path manager for the cutout directory.
    dia_source_id : `int`
        DiaSourceId whose ``{id}.png`` is checked.
    """
    return cpath.exists(dia_source_id, f"{dia_source_id}.png")


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
                    bad_flag_list=None, match_radius=0.5,
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
    cutout_path1, cutout_path2 : `str`, optional
        Base path to store cutouts for sources unique to the datasets.
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
        Data frame of matched sources; the rows are sources from the first
        dataset, with two columns added: ``src2_diaSourceId`` pointing to
        the matched diaSourceId in the second dataset, and
        ``xmatch_dist_arcsec`` giving the on-sky separation in arcseconds.
    """

    if make_cutouts and (cutout_path1 is None or cutout_path2 is None):
        errstr = ('You must supply a value for `cutout_path1` and `cutout_path2` if `make_cutouts` is True.')
        raise ValueError(errstr)

    if bad_flag_list is not None:
        # Snapshot and restore so we don't leave the caller's queries with a
        # different exclusion list than they started with.
        saved_flags1 = list(query1.diaSource_flags_exclude)
        saved_flags2 = list(query2.diaSource_flags_exclude)
        query1.set_excluded_diaSource_flags(bad_flag_list)
        query2.set_excluded_diaSource_flags(bad_flag_list)
        try:
            goodSrc1 = query1.load_sources(exclude_flagged=True)
            goodSrc2 = query2.load_sources(exclude_flagged=True)
        finally:
            query1.set_excluded_diaSource_flags(saved_flags1)
            query2.set_excluded_diaSource_flags(saved_flags2)
    else:
        goodSrc1 = query1.load_sources(exclude_flagged=True)
        goodSrc2 = query2.load_sources(exclude_flagged=True)

    if 'reliability' not in goodSrc1.columns:
        goodSrc1['reliability'] = None
    if 'reliability' not in goodSrc2.columns:
        goodSrc2['reliability'] = None

    # Cross-match within each (visit, detector) group.
    matched, unique1, unique2 = match_catalogs(
        goodSrc1, goodSrc2,
        radius=match_radius * u.arcsec,
        on=("visit", "detector"),
    )
    # Preserve the legacy column name on the returned `matched` DataFrame.
    matched = matched.rename(columns={"diaSourceId_2": "src2_diaSourceId"})

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

        # First figure out which cutouts already exist at the output path.
        # Series.apply passes one positional argument (the diaSourceId), but
        # _cutout_exists also needs the per-dataset cpath; partial binds it.
        unique1['pathexists'] = unique1['diaSourceId'].apply(
            functools.partial(_cutout_exists, cpath1))
        pathchk1 = unique1.loc[~unique1['pathexists']]

        unique2['pathexists'] = unique2['diaSourceId'].apply(
            functools.partial(_cutout_exists, cpath2))
        pathchk2 = unique2.loc[~unique2['pathexists']]

        # Only write those that don't exist yet
        plotter1.write_images(pathchk1, butler=butler1, njobs=njobs)
        plotter2.write_images(pathchk2, butler=butler2, njobs=njobs)

        if display_cutouts:
            for isrc in unique1.itertuples():
                fpath = cpath1(int(isrc.diaSourceId), f"{int(isrc.diaSourceId)}.png")
                print('Unique to dataset 1: {}'.format(int(isrc.diaSourceId)))
                display(Image(filename=fpath))

            for isrc in unique2.itertuples():
                fpath = cpath2(int(isrc.diaSourceId), f"{int(isrc.diaSourceId)}.png")

                print('Unique to dataset 2: {}'.format(int(isrc.diaSourceId)))
                display(Image(filename=fpath))

        # drop pathexists columns to return to original dataframe shape
        _ = unique1.pop('pathexists')
        _ = unique2.pop('pathexists')

    return unique1, unique2, matched


def compare_objects(query1, query2, match_radius=0.5):
    """Compare two APDB datasets by spatially crossmatching diaObjects.

    Parameters
    ----------
    query1 : `lsst.analysis.ap.DbQuery`
        DbQuery to first APDB (postgresql or slite file;
        NOT created in this function).
    query2 : `lsst.analysis.ap.DbQuery`
        DbQuery to second APDB (postgresql or slite file;
        NOT created in this function).
    match_radius : `double`
        Maximum allowable distance in arcsec between an object in
        data1 and data2.

    Returns
    -------
    unique1 : `pandas.DataFrame`
        Data frame of diaObjects only found in the first dataset.
    unique2 : `pandas.DataFrame`
        Data frame of diaObjects only found in the second dataset.
    matched : `pandas.DataFrame`
        Data frame of matched diaObjects; the rows are objects from the
        first dataset, with two columns added: ``obj2_diaObjectId``
        pointing to the matched diaObjectId in the second dataset, and
        ``xmatch_dist_arcsec`` giving the on-sky separation in arcseconds.
    """
    obj1 = query1.load_objects()
    obj2 = query2.load_objects()

    # diaObjects aren't tied to a single (visit, detector); match across
    # the full catalog with `on=()`.
    matched, unique1, unique2 = match_catalogs(
        obj1, obj2,
        radius=match_radius * u.arcsec,
        on=(),
        id_col="diaObjectId",
    )
    matched = matched.rename(columns={"diaObjectId_2": "obj2_diaObjectId"})

    print("{} matched objects; {} unique to set 1; {} unique to set 2.".format(
        len(matched), len(unique1), len(unique2)))

    return unique1, unique2, matched


def find_objects_sharing_sources(diaObjectId, sources1, sources2,
                                 objects1, objects2,
                                 max_distance_arcsec=2):
    """For a diaObjectId in run 1, return the full association cluster
    of diaSources and diaObjects from both runs.

    Treats the (diaSource, run-1-diaObject, run-2-diaObject) links as a
    graph -- each diaSource is connected to its owning diaObject in
    each run -- and grows the connected component reachable from the
    input diaObjectId until no new sources or objects are discovered.
    Catches arbitrarily deep merge/split chains across the two runs
    (e.g. run 2 merges A+B into Z, then a third source in B is split
    into a fourth object in run 2, etc.).

    Assumes diaSourceIds are stable across the two runs.

    Parameters
    ----------
    diaObjectId : `int`
        A diaObjectId, typically from `unique1` returned by
        `compare_objects`.
    sources1, sources2 : `pandas.DataFrame`
        Full diaSources catalogs from runs 1 and 2 (e.g. from
        ``query.load_sources()``). Each must contain `diaSourceId`,
        `diaObjectId`, `ra`, and `dec` columns.
    objects1, objects2 : `pandas.DataFrame`
        Full diaObjects catalogs from runs 1 and 2 (e.g. from
        ``query.load_objects()``). Each must contain `diaObjectId`,
        `ra`, and `dec` columns.
    max_distance_arcsec : `float`, optional
        If given, only include diaSources within this distance of the input
        diaObject's (ra, dec) in the search. All diaSources of the final
        diaObjects will still be returned, even if outside this distance.

    Returns
    -------
    sources : `pandas.DataFrame`
        Rows of `sources1` for every diaSource belonging to any of the
        found run-2 diaObjects (in run 2's view).
    related_objects1 : `pandas.DataFrame`
        Rows of `objects1` for every run-1 diaObject containing any of
        those diaSources in run 1.
    related_objects2 : `pandas.DataFrame`
        Rows of `objects2` for every run-2 diaObject containing any of
        those diaSources in run 2.
    """
    if max_distance_arcsec is not None:
        ref_match = objects1[objects1["diaObjectId"] == diaObjectId]
        if len(ref_match) == 0:
            raise ValueError(
                f"diaObjectId={diaObjectId} not found in objects1")
        ref_row = ref_match.iloc[0]
        ref = coord.SkyCoord(ra=ref_row["ra"] * u.deg,
                             dec=ref_row["dec"] * u.deg)
        # diaSourceIds (and their sky positions) match across runs, so
        # filtering once against sources1 suffices.
        sep = ref.separation(
            coord.SkyCoord(ra=sources1["ra"].values * u.deg,
                           dec=sources1["dec"].values * u.deg)
        ).to_value(u.arcsec)
        allowed_src_ids = set(
            sources1.loc[sep <= max_distance_arcsec, "diaSourceId"])
    else:
        allowed_src_ids = None

    # Breadth-first search for the connected component:
    # alternately expand sources from the currently-known objects,
    # then expand objects from the sources.
    # Terminates because every iteration adds at least one source
    # before the fixed-point check fires, and the source pool is finite.
    src_ids = set()
    obj1_ids = {diaObjectId}
    obj2_ids = set()

    while True:
        new_src_ids = set(
            sources1.loc[sources1["diaObjectId"].isin(obj1_ids),
                         "diaSourceId"])
        new_src_ids.update(
            sources2.loc[sources2["diaObjectId"].isin(obj2_ids),
                         "diaSourceId"])
        if allowed_src_ids is not None:
            new_src_ids &= allowed_src_ids
        if new_src_ids <= src_ids:
            break
        src_ids |= new_src_ids
        obj1_ids |= set(
            sources1.loc[sources1["diaSourceId"].isin(src_ids),
                         "diaObjectId"])
        obj2_ids |= set(
            sources2.loc[sources2["diaSourceId"].isin(src_ids),
                         "diaObjectId"])

    # Expand the final diaSource list to every source owned by any
    # surviving diaObject.
    final_src_ids = set(
        sources1.loc[sources1["diaObjectId"].isin(obj1_ids), "diaSourceId"])
    final_src_ids |= set(
        sources2.loc[sources2["diaObjectId"].isin(obj2_ids), "diaSourceId"])

    sources = sources1[sources1["diaSourceId"].isin(final_src_ids)]
    related_objects1 = objects1[objects1["diaObjectId"].isin(obj1_ids)]
    related_objects2 = objects2[objects2["diaObjectId"].isin(obj2_ids)]

    return sources, related_objects1, related_objects2


class _UnionFind:
    """Disjoint-set with path compression and union-by-rank.

    Used by `classify_association_clusters` to quickly find connected
    components of the (run-1 diaObject, run-2 diaObject) graph.
    """

    def __init__(self):
        self._parent = {}
        self._rank = {}

    def add(self, x):
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0

    def find(self, x):
        # Two-pass iterative find with path compression.
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1


def classify_association_clusters(sources1, sources2):
    """Enumerate and classify every association-disagreement cluster
    between two APDBs that share input diaSources.

    Builds the bipartite graph whose edges are
    ``(diaSource -> its run-1 diaObject, diaSource -> its run-2
    diaObject)`` over all common diaSources, runs union-find over the
    diaObjectIds to extract every connected component, and labels each
    cluster:

      * ``matched``  -- one run-1 obj <-> one run-2 obj.
      * ``split``    -- one run-1 obj split into multiple run-2 objs.
      * ``merged``   -- multiple run-1 objs merged into one run-2 obj.
      * ``tangled``  -- M run-1 objs <-> N run-2 objs, both > 1.

    Assumes diaSourceIds are stable across the two runs; diaSources
    present in only one catalog are silently skipped via inner join.

    Parameters
    ----------
    sources1, sources2 : `pandas.DataFrame`
        Full diaSources catalogs from runs 1 and 2 (e.g. from
        ``query.load_sources()``). Each must contain `diaSourceId`,
        `diaObjectId`, `ra`, and `dec` columns.

    Returns
    -------
    clusters : `pandas.DataFrame`
        One row per cluster, with columns:
          - ``kind``: matched / split / merged / tangled.
          - ``n_obj1``, ``n_obj2``: distinct diaObject counts per run.
          - ``n_sources``: distinct diaSources in the cluster.
          - ``obj1_ids``, ``obj2_ids``: tuples of diaObjectIds.
          - ``ra``, ``dec``: mean sky position of the cluster's
            diaSources (degrees).
    """
    # Pre-define the types so that value_counts() and groupby()
    # include unused kinds with a count of 0.
    kind_dtype = pd.CategoricalDtype(
        categories=["matched", "split", "merged", "tangled"], ordered=True)

    paired = sources1[["diaSourceId", "diaObjectId", "ra", "dec"]].merge(
        sources2[["diaSourceId", "diaObjectId"]].rename(
            columns={"diaObjectId": "diaObjectId_2"}),
        on="diaSourceId", how="inner")

    if len(paired) == 0:
        empty = pd.DataFrame(columns=[
            "kind", "n_obj1", "n_obj2", "n_sources",
            "obj1_ids", "obj2_ids", "ra", "dec"])
        empty["kind"] = empty["kind"].astype(kind_dtype)
        return empty

    # Define a namespace for the two runs so identical numeric ids in run 1 and
    # run 2 don't collide as keys.
    keys1 = [("r1", int(i)) for i in paired["diaObjectId"].to_numpy()]
    keys2 = [("r2", int(i)) for i in paired["diaObjectId_2"].to_numpy()]

    uf = _UnionFind()
    for k in set(keys1):
        uf.add(k)
    for k in set(keys2):
        uf.add(k)
    for k1, k2 in zip(keys1, keys2):
        uf.union(k1, k2)

    paired = paired.assign(_cluster=[uf.find(k) for k in keys1])

    rows = []
    for _, grp in paired.groupby("_cluster", sort=False):
        ids_a = tuple(sorted(int(i) for i in grp["diaObjectId"].unique()))
        ids_b = tuple(sorted(int(i) for i in grp["diaObjectId_2"].unique()))
        n1, n2 = len(ids_a), len(ids_b)
        if n1 == 1 and n2 == 1:
            kind = "matched"
        elif n1 == 1:
            kind = "split"
        elif n2 == 1:
            kind = "merged"
        else:
            kind = "tangled"
        rows.append({
            "kind": kind,
            "n_obj1": n1, "n_obj2": n2,
            "n_sources": grp["diaSourceId"].nunique(),
            "obj1_ids": ids_a, "obj2_ids": ids_b,
            "ra": float(grp["ra"].mean()),
            "dec": float(grp["dec"].mean()),
        })

    result = pd.DataFrame(rows)
    result["kind"] = result["kind"].astype(kind_dtype)
    return result


# Colors used by the cutout plotters to give each distinct
# diaObjectId its own marker color.
_OBJECT_PALETTE = ("lime", "red", "cyan", "magenta", "yellow", "orange",
                   "deepskyblue", "pink", "white", "violet", "gold",
                   "lightgreen")


def _prepare_object_overlays(objects, palette):
    """Deduplicate `objects` by diaObjectId and assign one palette color
    per distinct id, returning the parallel arrays the cutout renderer
    needs: ``(ids, ras, decs, colors)``. Done once per call so the same
    color identifies the same diaObject across every cutout.
    """
    obj_unique = objects.drop_duplicates(subset="diaObjectId")
    # Prefer the run-2 id when present (matched rows carry both); pandas
    # concat promotes the column to float64 if any rows lack it, so cast
    # back to int64 after filling.
    if "obj2_diaObjectId" in obj_unique.columns:
        obj_ids = obj_unique["obj2_diaObjectId"].combine_first(
            obj_unique["diaObjectId"]).astype(np.int64).to_numpy()
    else:
        obj_ids = obj_unique["diaObjectId"].astype(np.int64).to_numpy()
    obj_ras = np.asarray(obj_unique["ra"])
    obj_decs = np.asarray(obj_unique["dec"])
    obj_colors = [palette[i % len(palette)] for i in range(len(obj_ids))]
    return obj_ids, obj_ras, obj_decs, obj_colors


def _load_cutout(butler, row, *, size, image_type, image_datasets):
    """Fetch the requested image dataset for this row's (visit,
    detector) and return a small dict with everything the renderer
    needs: pixel data, dimensions, cutout origin, WCS, an
    ImageNormalize tuned to the central source, and the `image_type`
    label used in the cutout title.

    Loading is separated from rendering so callers that need to draw
    the same cutout into multiple Axes can pay the butler.get + getCutout
    cost once.
    """
    import astropy.visualization as aviz
    import lsst.geom

    dataset = image_datasets[image_type]
    data_id = {"visit": int(row.visit), "detector": int(row.detector)}
    exposure = butler.get(dataset, data_id)

    center = lsst.geom.SpherePoint(row.ra, row.dec, lsst.geom.degrees)
    extent = lsst.geom.Extent2I(size, size)
    cutout = exposure.getCutout(center, extent)
    data = cutout.image.array
    ny, nx = data.shape

    if image_type == "difference":
        # Normalize on a small central window so the source dominates
        # the dynamic range.
        cy, cx = ny // 2, nx // 2
        half = min(7, cy, cx)
        norm_data = data[cy - half:cy + half + 1, cx - half:cx + half + 1]
    else:
        norm_data = data
    norm = aviz.ImageNormalize(
        norm_data, interval=aviz.MinMaxInterval(),
        stretch=aviz.AsinhStretch(a=0.1))

    return {
        "data": data, "ny": ny, "nx": nx,
        "wcs": cutout.wcs,
        "x0": cutout.getX0(), "y0": cutout.getY0(),
        "norm": norm,
        "image_type": image_type,
    }


def _render_cutout_axes(ax, row, cutout_data, sources,
                        obj_ids, obj_ras, obj_decs, obj_colors, *,
                        marker_size, marker_symbol,
                        source_marker_size, current_source_marker_size,
                        current_source_color,
                        title=None, subtitle=""):
    """Render one diaSource cutout onto an existing matplotlib Axes
    using preloaded data from `_load_cutout`.

    Internal helper shared by `plot_cutouts_with_object_markers` and
    `plot_objects_sharing_sources`. The caller owns figure creation,
    layout, saving, and displaying.

    By default the axes title is built from `row` as
    ``"diaSourceId=... (image_type, visit=..., det=...)"``. Pass
    `title=` explicitly (including ``""``) to override or suppress
    that line -- useful when a parent figure or subfigure already
    carries the shared header. If `subtitle` is non-empty it is drawn
    on a second title line.
    """
    from matplotlib import cm

    data = cutout_data["data"]
    ny = cutout_data["ny"]
    nx = cutout_data["nx"]
    x0 = cutout_data["x0"]
    y0 = cutout_data["y0"]
    wcs = cutout_data["wcs"]
    norm = cutout_data["norm"]
    image_type = cutout_data["image_type"]

    ax.imshow(data, cmap=cm.bone, interpolation="none", norm=norm,
              origin="lower", aspect="equal",
              extent=(0, nx, 0, ny))

    this_id = int(row.diaSourceId)

    # Project every supplied diaSource into the cutout frame once,
    # then split into "this cutout's diaSource" vs every other
    # diaSource whose sky position falls inside this cutout
    # (regardless of which image it was detected on).
    if len(sources) > 0:
        src_xs, src_ys = wcs.skyToPixelArray(
            np.asarray(sources["ra"]),
            np.asarray(sources["dec"]),
            degrees=True)
        src_xs = src_xs - x0
        src_ys = src_ys - y0
        src_in_frame = (
            (src_xs >= 0) & (src_xs < nx)
            & (src_ys >= 0) & (src_ys < ny))
        id_arr = sources["diaSourceId"].to_numpy()
        other_src_mask = src_in_frame & (id_arr != this_id)
        current_src_mask = src_in_frame & (id_arr == this_id)
    else:
        src_xs = src_ys = None
        other_src_mask = current_src_mask = None

    if other_src_mask is not None and other_src_mask.any():
        # Share the current-diaSource color so the source positions are
        # easy to read against the cm.bone background; the marker
        # symbol (+ vs x) still distinguishes them.
        ax.scatter(src_xs[other_src_mask], src_ys[other_src_mask],
                   s=source_marker_size, marker="+",
                   c=current_source_color, linewidths=1.0,
                   label="other diaSource")

    if len(obj_ids) > 0:
        xs, ys = wcs.skyToPixelArray(obj_ras, obj_decs, degrees=True)
        xs = xs - x0
        ys = ys - y0
        in_bounds = (xs >= 0) & (xs < nx) & (ys >= 0) & (ys < ny)
    else:
        in_bounds = np.zeros(0, dtype=bool)

    for i in np.flatnonzero(in_bounds):
        ax.scatter(xs[i], ys[i],
                   s=marker_size, marker=marker_symbol,
                   facecolors="none", edgecolors=obj_colors[i],
                   linewidths=1.5,
                   label=f"diaObjectId={int(obj_ids[i])}")

    # Current diaSource last so it stays on top of any overlapping
    # diaObject marker at the cutout center.
    if current_src_mask is not None and current_src_mask.any():
        ax.scatter(src_xs[current_src_mask], src_ys[current_src_mask],
                   s=current_source_marker_size, marker="x",
                   c=current_source_color, linewidths=2.0,
                   label=f"current diaSourceId={this_id}")

    if title is None:
        title = (f"diaSourceId={this_id} "
                 f"({image_type}, visit={int(row.visit)}, "
                 f"det={int(row.detector)})")
    if subtitle:
        title = f"{title}\n{subtitle}" if title else subtitle
    if title:
        ax.set_title(title, fontsize="small")
    ax.set_xticks([])
    ax.set_yticks([])
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", fontsize="x-small", framealpha=0.7)


def plot_cutouts_with_object_markers(sources, butler, objects, *,
                                     output_path=None,
                                     display_cutouts=False,
                                     size=51,
                                     image_type="difference",
                                     image_datasets=_IMAGE_DATASETS,
                                     marker_size=80,
                                     marker_symbol="o",
                                     palette=_OBJECT_PALETTE,
                                     source_marker_size=80,
                                     current_source_marker_size=180,
                                     current_source_color="yellow"):
    """Plot per-diaSource cutouts with overlaid markers at given diaObject
    sky positions.

    For each diaSource in `sources`, fetch a square cutout from `butler`
    centered on the source's (ra, dec). On each cutout draw:

      * A small ``+`` marker at every other diaSource in `sources`
        whose sky position lands inside the cutout, regardless of which
        (visit, detector) it was detected on.
      * A distinct ``x`` marker for the diaSource the cutout is
        centered on (the "current" diaSource).
      * One color-coded marker per distinct diaObjectId in `objects`,
        cycling through `palette`; the same color identifies the same
        diaObject across every cutout in the run.

    Markers that fall outside the cutout bounds are skipped.

    Typical use: visualize how a group of diaSources (all originally
    associated with one diaObjectId in run 1) got redistributed across
    diaObjects in run 2. `sources` and `objects` are usually built from
    the output of `find_objects_sharing_sources`::

        sources, ro1, ro2 = find_objects_sharing_sources(
            diaObjectId, sources1, sources2, objects1, objects2)
        objects = pd.concat([ro1, ro2])
        plot_cutouts_with_object_markers(
            sources, butler1, objects, display_cutouts=True,
        )

    Parameters
    ----------
    sources : `pandas.DataFrame`
        DiaSources to cut out. Must contain `diaSourceId`, `ra`, `dec`,
        `visit`, and `detector` columns.
    butler : `lsst.daf.butler.Butler`
        Butler containing the image datasets for these (visit, detector)
        pairs.
    objects : `pandas.DataFrame`
        DiaObjects to mark. Must contain `diaObjectId`, `ra`, and `dec`
        columns. Duplicate diaObjectIds are dropped (first row wins).
        If an `obj2_diaObjectId` column is present (e.g. for rows from
        a `matched` DataFrame returned by `compare_objects`), the run-2
        id is shown in the legend in preference to the run-1
        `diaObjectId`.
    output_path : `str`, optional
        Directory to write ``{diaSourceId}.png`` files to. Created if
        missing. Pass None to skip writing.
    display_cutouts : `bool`, optional
        If True, display each cutout inline (notebook).
    size : `int`, optional
        Cutout side length in pixels.
    image_type : {"science", "template", "difference"}, optional
        Which image to render.
    image_datasets : `dict` [`str`, `str`], optional
        Mapping from image-type key to butler dataset name.
    marker_size : `int`, optional
        matplotlib scatter ``s`` parameter for diaObject markers.
    marker_symbol : `str`, optional
        matplotlib scatter ``marker`` parameter for diaObject markers.
    palette : sequence of `str`, optional
        Color cycle used to assign one color per diaObjectId.
    source_marker_size : `int`, optional
        Scatter ``s`` parameter for the small ``+`` markers drawn at
        the positions of the *other* diaSources in `sources`.
    current_source_marker_size : `int`, optional
        Scatter ``s`` parameter for the distinct marker drawn at the
        diaSource the cutout is centered on.
    current_source_color : `str`, optional
        Color of the current-diaSource marker.
    """
    import matplotlib.pyplot as plt

    if image_type not in image_datasets:
        raise ValueError(
            f"image_type must be one of {sorted(image_datasets)}, "
            f"got {image_type!r}")

    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)

    overlays = _prepare_object_overlays(objects, palette)

    for row in sources.itertuples(index=False):
        cutout_data = _load_cutout(butler, row, size=size,
                                   image_type=image_type,
                                   image_datasets=image_datasets)
        fig, ax = plt.subplots()
        _render_cutout_axes(
            ax, row, cutout_data, sources, *overlays,
            marker_size=marker_size, marker_symbol=marker_symbol,
            source_marker_size=source_marker_size,
            current_source_marker_size=current_source_marker_size,
            current_source_color=current_source_color)

        if output_path is not None:
            fpath = os.path.join(output_path, f"{int(row.diaSourceId)}.png")
            fig.savefig(fpath, bbox_inches="tight")
        if display_cutouts:
            display(fig)
        plt.close(fig)


def plot_objects_sharing_sources(diaObjectId, sources1, sources2,
                                 objects1, objects2, butler, *,
                                 max_distance_arcsec=None,
                                 output_path=None,
                                 display_figure=True,
                                 column_labels=("run 1", "run 2"),
                                 figsize_per_row=4.0,
                                 size=51,
                                 image_type="difference",
                                 image_datasets=_IMAGE_DATASETS,
                                 marker_size=80,
                                 marker_symbol="o",
                                 palette=_OBJECT_PALETTE,
                                 source_marker_size=80,
                                 current_source_marker_size=180,
                                 current_source_color="yellow"):
    """Two-column cutout figure comparing the run-1 and run-2 views of
    an association cluster.

    Calls `find_objects_sharing_sources` internally to identify the
    cluster of diaSources and diaObjects reachable from the input
    `diaObjectId`, then renders one row per diaSource in the cluster.
    The same cutout image is loaded once per row and drawn into both
    columns: the left panel is overlaid with run-1 diaObject markers
    (from `objects1`) and the right panel with run-2 diaObject markers
    (from `objects2`). Each column's diaObjects get their own palette
    mapping, so the same color in the left and right columns does
    *not* imply the same diaObject.

    Parameters
    ----------
    diaObjectId : `int`
        Starting diaObjectId for the cluster walk.
    sources1, sources2, objects1, objects2 : `pandas.DataFrame`
        Forwarded to `find_objects_sharing_sources`.
    butler : `lsst.daf.butler.Butler`
        Butler used to fetch the cutout images. The same image backs
        both panels of a given row.
    max_distance_arcsec : `float`, optional
        Forwarded to `find_objects_sharing_sources`.
    output_path : `str`, optional
        Filename to write the combined figure to as a PNG. Parent
        directories are created if missing.
    display_figure : `bool`, optional
        If True, display the figure inline (notebook).
    column_labels : pair of `str`, optional
        Labels appended to each cutout's title to identify the column.
    figsize_per_row : `float`, optional
        Height in inches allocated to each cutout row.
    All other kwargs:
        Forwarded to the cutout renderer; same meaning as in
        `plot_cutouts_with_object_markers`.

    Returns
    -------
    sources, related_objects1, related_objects2 : `pandas.DataFrame`
        The catalogs returned by `find_objects_sharing_sources`.
    """
    import matplotlib.pyplot as plt

    if image_type not in image_datasets:
        raise ValueError(
            f"image_type must be one of {sorted(image_datasets)}, "
            f"got {image_type!r}")

    sources, ro1, ro2 = find_objects_sharing_sources(
        diaObjectId, sources1, sources2, objects1, objects2,
        max_distance_arcsec=max_distance_arcsec)

    if len(sources) == 0:
        print(f"No diaSources in the cluster for "
              f"diaObjectId={diaObjectId}")
        return sources, ro1, ro2

    left_overlays = _prepare_object_overlays(ro1, palette)
    right_overlays = _prepare_object_overlays(ro2, palette)

    n_rows = len(sources)
    # One subfigure per row so each row can carry a single shared
    # suptitle above both panels; the per-axes title is then just the
    # column label.
    fig = plt.figure(figsize=(8, figsize_per_row * n_rows),
                     constrained_layout=True)
    subfigs = np.atleast_1d(fig.subfigures(n_rows, 1, squeeze=False).ravel())

    common_kw = dict(
        marker_size=marker_size, marker_symbol=marker_symbol,
        source_marker_size=source_marker_size,
        current_source_marker_size=current_source_marker_size,
        current_source_color=current_source_color)

    for i, row in enumerate(sources.itertuples(index=False)):
        sf = subfigs[i]
        sf.suptitle(
            f"diaSourceId={int(row.diaSourceId)} "
            f"({image_type}, visit={int(row.visit)}, "
            f"det={int(row.detector)})",
            fontsize="small")
        ax_left, ax_right = sf.subplots(1, 2)
        # Load once; both panels in this row use the same image.
        cutout_data = _load_cutout(butler, row, size=size,
                                   image_type=image_type,
                                   image_datasets=image_datasets)
        _render_cutout_axes(ax_left, row, cutout_data, sources,
                            *left_overlays,
                            title="", subtitle=column_labels[0],
                            **common_kw)
        _render_cutout_axes(ax_right, row, cutout_data, sources,
                            *right_overlays,
                            title="", subtitle=column_labels[1],
                            **common_kw)

    if output_path is not None:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    if display_figure:
        display(fig)
    plt.close(fig)

    return sources, ro1, ro2


def get_xy_from_source_table(table, wcs, degrees=None):
    """Convert ra/dec coordinates in an astropy table/pandas data frame to
    pixel x/y positions.
    """
    try:
        ra = table['ra']
        dec = table['dec']
        inferred_degrees = True
    except KeyError:
        ra = table['coord_ra']
        dec = table['coord_dec']
        inferred_degrees = False
    if degrees is None:
        degrees = inferred_degrees

    x, y = wcs.skyToPixelArray(ra, dec, degrees=degrees)
    return astropy.table.Table.from_pandas(pd.DataFrame({'x': x, 'y': y}))


# Palette used by the `color_by` flag-bucketing mode. Sources with none of
# the requested flags set get the residual color "white", which is kept out
# of this palette so it never collides with a flagged bucket.
_FLAG_PALETTE = ("red", "orange", "yellow", "magenta", "cyan", "green")


def _group_sources_by_flag(table, flag_names, palette=_FLAG_PALETTE):
    """Split a source table into per-flag buckets for color-coded overlay.

    Each row is assigned to the first flag in ``flag_names`` whose column
    is True; remaining rows go into a residual "no flag" bucket. Names that
    aren't present as columns in ``table`` are silently skipped.

    Parameters
    ----------
    table : table-like
        Anything that supports ``len(table)``, ``table[name]`` returning a
        boolean-coercible column, and ``table[bool_array]`` row selection.
    flag_names : sequence of str
        Column names to group on. Order determines color *and* priority
        when a row has multiple flags set.
    palette : sequence of str, optional
        Cycle of display ``ctype`` values to assign in order.

    Returns
    -------
    buckets : list of ``(subset_table, ctype, legend)`` tuples.
    """
    n = len(table)
    if n == 0:
        return []
    remaining = np.ones(n, dtype=bool)
    buckets = []
    for i, flag in enumerate(flag_names):
        try:
            col = table[flag]
        except KeyError:
            continue
        mask = np.asarray(col, dtype=bool) & remaining
        if mask.any():
            buckets.append((table[mask], palette[i % len(palette)], flag))
            remaining = remaining & ~mask
    if remaining.any():
        buckets.append((table[remaining], "white", "no flag"))
    return buckets


def _collect_overlays(butler, data_id, wcs, *,
                      reliability_threshold,
                      show_unfiltered, show_trailed,
                      show_rejected, show_marginal, show_solar_system,
                      show_apdb, show_reliability_labels,
                      color_by):
    """Load catalogs from one butler and build the overlay record list.

    Shared between `display_images` and `display_images_ab`. Catalogs that
    aren't present for this dataId are silently skipped.

    Returns
    -------
    overlays : list of ``(x_arr, y_arr, symbol, size, ctype, legend)`` tuples.
    reliability_labels : dict or None
        ``{"x", "y", "reliability"}`` arrays for the good APDB diaSources,
        suitable for drawing text annotations next to each marker.
    solar_system_labels : dict or None
        ``{"x", "y", "designation"}`` arrays for matched solar-system
        sources, suitable for drawing the designation as text next to each
        marker.
    """
    def _try_get(dataset):
        try:
            return butler.get(dataset, data_id)
        except DatasetNotFoundError:
            return None

    overlays = []

    def _add(table, *, symbol, size, ctype, legend, use_radec=True):
        if table is None or len(table) == 0:
            return
        if use_radec:
            xy = get_xy_from_source_table(table, wcs)
            x_arr = xy["x"].data
            y_arr = xy["y"].data
        else:
            x_arr = table["x"].data
            y_arr = table["y"].data
        overlays.append((x_arr, y_arr, symbol, size, ctype, legend))

    if show_unfiltered:
        unfiltered = _try_get("dia_source_unfiltered")
        if unfiltered is not None and len(unfiltered) > 0:
            non_sky = unfiltered[~unfiltered["sky_source"]]
            if color_by:
                for sub, ctype, flag in _group_sources_by_flag(non_sky, color_by):
                    _add(sub, symbol="+", size=10, ctype=ctype,
                         legend=f"unfiltered: {flag}")
            else:
                _add(non_sky, symbol="+", size=10, ctype="red",
                     legend="unfiltered candidate")

    if show_trailed:
        _add(_try_get("long_trailed_source_detector"),
             symbol="x", size=30, ctype="magenta", legend="long-trailed source")
    if show_rejected:
        _add(_try_get("rejected_dia_source"),
             symbol="+", size=10, ctype="orange", legend="rejected diaSource")
    if show_marginal:
        _add(_try_get("marginal_new_dia_source"),
             symbol="+", size=10, ctype="yellow", legend="marginal new diaSource")

    # Load dia_source_apdb once: it backs the APDB reliability overlay and
    # also supplies pixel x/y for the solar-system overlay (ss_source_detector
    # carries only the matched diaSourceId, not coordinates).
    dia_apdb = None
    if show_solar_system or show_apdb:
        dia_apdb = _try_get("dia_source_apdb")

    solar_system_labels = None
    if show_solar_system:
        ss = _try_get("ss_source_detector")
        if (ss is not None and len(ss) > 0
                and dia_apdb is not None and len(dia_apdb) > 0):
            # ss_source_detector lacks coords; match each diaSourceId to
            # the APDB row to recover its pixel x/y.
            ss_ids = np.asarray(ss["diaSourceId"])
            apdb_ids = np.asarray(dia_apdb["diaSourceId"])
            idx_in_apdb = pd.Series(np.arange(len(apdb_ids)), index=apdb_ids).reindex(ss_ids)
            keep = idx_in_apdb.notna().to_numpy()
            if keep.any():
                apdb_idx = idx_in_apdb.dropna().astype(int).to_numpy()
                x_arr = np.asarray(dia_apdb["x"])[apdb_idx]
                y_arr = np.asarray(dia_apdb["y"])[apdb_idx]
                designation = np.asarray(ss["designation"])[keep]
                overlays.append((x_arr, y_arr, "o", 12, "cyan", "solar-system match"))
                solar_system_labels = {"x": x_arr, "y": y_arr, "designation": designation, }

    reliability_labels = None
    if show_apdb:
        if dia_apdb is not None and len(dia_apdb) > 0:
            good_mask = dia_apdb["reliability"] > reliability_threshold
            good_src = dia_apdb[good_mask]
            bad_src = dia_apdb[~good_mask]
            _add(good_src, symbol="o", size=12, ctype="blue", use_radec=False,
                 legend=f"APDB, reliability > {reliability_threshold:g}")
            _add(bad_src, symbol="o", size=12, ctype="red", use_radec=False,
                 legend=f"APDB, reliability <= {reliability_threshold:g}")
            if show_reliability_labels and len(good_src) > 0:
                reliability_labels = {
                    "x": good_src["x"].data,
                    "y": good_src["y"].data,
                    "reliability": good_src["reliability"],
                }

    return overlays, reliability_labels, solar_system_labels


def _print_overlay_legend(overlays, header, indent=""):
    """Print a one-line-per-overlay legend for a single panel."""
    print(f"{indent}{header}")
    for x_arr, _, symbol, _, ctype, legend in overlays:
        print(f"{indent}  {len(x_arr):5d}  {ctype:>8s} {symbol}  {legend}")


def _draw_overlays_on_current_frame(afw_display, overlays,
                                    reliability_labels, solar_system_labels,
                                    label_size=3):
    """Stamp one set of overlays + optional reliability and solar-system
    designation labels onto the active frame.

    ``label_size`` is the text size (in pixels) used for both label sets.
    """
    # Scale the text offset with the size so larger labels still clear the
    # circle markers they annotate.
    label_offset = max(14, 2 * label_size)
    with afw_display.Buffering():
        for x_arr, y_arr, symbol, size, ctype, _ in overlays:
            for x, y in zip(x_arr, y_arr):
                afw_display.dot(symbol, x, y, size=size, ctype=ctype)
        if reliability_labels is not None:
            # Offset the score text so it doesn't sit on top of the marker.
            for r, x, y in zip(reliability_labels["reliability"],
                               reliability_labels["x"],
                               reliability_labels["y"]):
                afw_display.dot(f"{r:.2f}", x + label_offset, y,
                                size=label_size, ctype="cyan")
        if solar_system_labels is not None:
            # Offset SS designations *below* the marker so they don't
            # overplot any reliability score drawn to the right.
            for desig, x, y in zip(solar_system_labels["designation"],
                                   solar_system_labels["x"],
                                   solar_system_labels["y"]):
                afw_display.dot(str(desig), x, y + label_offset,
                                size=label_size, ctype="cyan")


def _strip_ds9_metadata(*exposures):
    """Drop LTV1/LTV2 keys from each exposure's metadata in place."""
    for exp in exposures:
        md = exp.metadata
        for k in ("LTV1", "LTV2"):
            if md.exists(k):
                md.remove(k)


def display_images(butler, visit, detector, backend="firefly", *,
                   reliability_threshold=0.1,
                   show_unfiltered=True,
                   show_trailed=True,
                   show_rejected=True,
                   show_marginal=True,
                   show_solar_system=True,
                   show_apdb=True,
                   show_reliability_labels=True,
                   label_size=3,
                   color_by=None,
                   mask_transparency=80,
                   strip_metadata=True,
                   image_datasets=_IMAGE_DATASETS):
    """Display the science, template, and difference images for a given
    visit+detector with diagnostic catalog markers overlaid.

    Three frames are produced (science, template, difference) and the same
    overlays are drawn on each. Catalogs that are missing from the butler
    are silently skipped, so the same call works against partial outputs.

    Default overlay key:

    ============================  =======  ==========================
    catalog                       symbol   color
    ============================  =======  ==========================
    unfiltered candidates         ``+``    red
    long-trailed sources          ``x``    magenta
    rejected diaSources           ``+``    orange
    marginal new diaSources       ``+``    yellow
    solar-system matches          ``o``    cyan
    APDB, reliability > threshold ``o``    blue (+ score text)
    APDB, reliability ≤ threshold ``o``    red
    ============================  =======  ==========================

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        Butler to load data from.
    visit, detector : `int`
        Visit and detector ids to load data for.
    backend : `str`, optional
        afw display backend (typically "firefly" or "ds9").
    reliability_threshold : `float`, optional
        APDB diaSources with reliability strictly greater than this are
        drawn as "good" (blue); the rest as "bad" (red).
    show_unfiltered, show_trailed, show_rejected, show_marginal,
    show_solar_system, show_apdb : `bool`, optional
        Toggle individual catalog overlays.
    show_reliability_labels : `bool`, optional
        If True, annotate each good APDB diaSource with its reliability score.
    label_size : `int`, optional
        Text size (in pixels) for the reliability score and solar-system
        designation annotations.
    color_by : sequence of `str`, optional
        Flag column names from ``dia_source_unfiltered``. When supplied,
        the unfiltered-candidate overlay is split into buckets colored by
        which named flag fires first (list order = color *and* priority),
        with a residual white bucket for rows that match none of them.
        Unknown column names are silently skipped. Example::

            color_by=["pixelFlags_bad", "pixelFlags_edge",
                      "ip_diffim_DipoleFit_classification",
                      "pixelFlags_saturated"]
    mask_transparency : `int` or `None`, optional
        Mask-plane transparency forwarded to the display (0 = opaque,
        100 = fully transparent). Pass ``None`` to leave the backend's
        current setting untouched.
    strip_metadata : `bool`, optional
        Drop ``LTV1``/``LTV2`` keywords from each exposure's metadata
        before sending to the backend. Needed for ds9 to align frames.
    image_datasets : `dict` [`str`, `str`], optional
        Mapping from image-type key (``"science"``, ``"template"``,
        ``"difference"``) to butler dataset name. Override to point at
        alternate dataset types.
    """
    data_id = {"visit": visit, "detector": detector}

    diffim = butler.get(image_datasets["difference"], data_id)
    science = butler.get(image_datasets["science"], data_id)
    template = butler.get(image_datasets["template"], data_id)
    template = template[science.getBBox()]
    if strip_metadata:
        _strip_ds9_metadata(science, diffim, template)
    images = {"science": science, "template": template, "difference": diffim}

    overlays, reliability_labels, solar_system_labels = _collect_overlays(
        butler, data_id, diffim.wcs,
        reliability_threshold=reliability_threshold,
        show_unfiltered=show_unfiltered,
        show_trailed=show_trailed, show_rejected=show_rejected,
        show_marginal=show_marginal, show_solar_system=show_solar_system,
        show_apdb=show_apdb,
        show_reliability_labels=show_reliability_labels,
        color_by=color_by,
    )
    _print_overlay_legend(
        overlays, f"visit={visit}, detector={detector} -- overlay legend:")

    afw_display = lsst.afw.display.Display(backend=backend)
    if mask_transparency is not None:
        afw_display.setMaskTransparency(mask_transparency)
    for frame, image_name in enumerate(("science", "template", "difference")):
        afw_display.frame = frame
        afw_display.image(images[image_name], title=image_name)
        _draw_overlays_on_current_frame(
            afw_display, overlays, reliability_labels, solar_system_labels,
            label_size=label_size)

    try:
        afw_display.alignImages(match_type="Pixel")
    except NotImplementedError:
        print(f"WARNING: cannot automatically align and lock images with backend={backend!r}.")


def display_images_ab(butler_a, butler_b, visit, detector, *,
                      image_type="difference",
                      labels=("A", "B"),
                      backend="firefly",
                      reliability_threshold=0.1,
                      show_unfiltered=True,
                      show_trailed=True,
                      show_rejected=True,
                      show_marginal=True,
                      show_solar_system=True,
                      show_apdb=True,
                      show_reliability_labels=True,
                      label_size=3,
                      color_by=None,
                      mask_transparency=80,
                      strip_metadata=True,
                      image_datasets=_IMAGE_DATASETS):
    """Display one image type side-by-side from two butlers, with overlays.

    Loads the same (visit, detector) from ``butler_a`` and ``butler_b``,
    places them in two frames, and draws each butler's catalog overlays on
    its own frame. Intended for A/B-testing pipeline-config changes that affect
    detection or subtraction quality.

    Parameters
    ----------
    butler_a, butler_b : `lsst.daf.butler.Butler`
        Two butlers, typically from different pipeline runs of the same data.
    visit, detector : `int`
        Visit and detector ids to load data for.
    image_type : {"science", "template", "difference"}, optional
        Which image dataset to compare. Default ``"difference"``.
    labels : pair of `str`, optional
        Short tags for the two frames; appear in the image title and the
        legend header. Default ``("A", "B")``.
    backend : `str`, optional
        afw display backend (typically "firefly" or "ds9").
    reliability_threshold, show_unfiltered, show_trailed, show_rejected,
    show_marginal, show_solar_system, show_apdb, show_reliability_labels,
    label_size, color_by, mask_transparency, strip_metadata, image_datasets
        Same meaning as in `display_images`. Applied to overlays from
        *both* butlers.
    """
    if image_type not in image_datasets:
        raise ValueError(
            f"image_type must be one of {sorted(image_datasets)}, got {image_type!r}")
    dataset = image_datasets[image_type]
    data_id = {"visit": visit, "detector": detector}

    image_a = butler_a.get(dataset, data_id)
    image_b = butler_b.get(dataset, data_id)
    if image_type == "template":
        # Templates are usually larger than the science footprint; clip them
        # to the science bbox so the two frames have matching extents.
        sci_a = butler_a.get(image_datasets["science"], data_id)
        sci_b = butler_b.get(image_datasets["science"], data_id)
        image_a = image_a[sci_a.getBBox()]
        image_b = image_b[sci_b.getBBox()]
    if strip_metadata:
        _strip_ds9_metadata(image_a, image_b)

    common = dict(
        reliability_threshold=reliability_threshold,
        show_unfiltered=show_unfiltered,
        show_trailed=show_trailed, show_rejected=show_rejected,
        show_marginal=show_marginal, show_solar_system=show_solar_system,
        show_apdb=show_apdb, show_reliability_labels=show_reliability_labels,
        color_by=color_by,
    )
    overlays_a, rel_a, ss_a = _collect_overlays(butler_a, data_id, image_a.wcs, **common)
    overlays_b, rel_b, ss_b = _collect_overlays(butler_b, data_id, image_b.wcs, **common)

    label_a, label_b = labels
    print(f"visit={visit}, detector={detector}: A/B comparison of {image_type!r}")
    _print_overlay_legend(overlays_a, f"-- {label_a} overlay legend:", indent="  ")
    _print_overlay_legend(overlays_b, f"-- {label_b} overlay legend:", indent="  ")

    afw_display = lsst.afw.display.Display(backend=backend)
    if mask_transparency is not None:
        afw_display.setMaskTransparency(mask_transparency)
    for frame, (tag, image, overlays, rel, ss) in enumerate((
            (label_a, image_a, overlays_a, rel_a, ss_a),
            (label_b, image_b, overlays_b, rel_b, ss_b))):
        afw_display.frame = frame
        afw_display.image(image, title=f"{image_type} ({tag})")
        _draw_overlays_on_current_frame(afw_display, overlays, rel, ss,
                                        label_size=label_size)

    try:
        afw_display.alignImages(match_type="Pixel")
    except NotImplementedError:
        print(f"WARNING: cannot automatically align and lock images with backend={backend!r}.")


def extract_timestamped_messages(log: str | dict[str, Any]) -> str:
    """
    Extract records[*].(asctime, message) from an LSST-style JSON log and
    format as:
        2026-02-25T04:15:35.092108Z Preparing execution...
    one per line.

    Parameters
    ----------
    log:
        Either the JSON text (str) or a parsed dict.
    sort:
        If True, sort by asctime (robust if log fragments are concatenated).

    Returns
    -------
    str
        Joined lines.
    """
    if isinstance(log, str):
        s = log.strip()

        # Handle the case where the *JSON itself* is wrapped in quotes, like:
        # '"{...}"' or "'{...}'"
        if (len(s) >= 2) and (s[0] == s[-1]) and s[0] in ("'", '"'):
            s = s[1:-1]

        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            # One more attempt: sometimes a quoted-JSON string is itself
            # JSON-encoded e.g. "\"{...}\""
            obj = json.loads(json.loads(s))
    else:
        obj = log

    records = obj.get("records", [])
    if not isinstance(records, list):
        raise TypeError("Expected obj['records'] to be a list.")

    rows: list[tuple[datetime, str, str]] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        ts = rec.get("asctime")
        msg = rec.get("message")
        if not ts or msg is None:
            continue

        # Parse ISO-8601 with trailing "Z"
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
        rows.append((dt, ts, str(msg)))

    return "\n".join(f"{ts} {msg}" for _, ts, msg in rows)

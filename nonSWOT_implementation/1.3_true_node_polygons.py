#!/usr/bin/env python
"""Build node footprints from RiverObs's pixel assignment.

Usage:
    python 1.3_true_node_polygons.py \\
        /path/rasters/upperYR_071024_water_3m.tif \\
        /path/riverobs_out/v17b/ortho_upperYR_071024_v17b_pixcvec.nc \\
        /path/out/true_node_polygons_upperYR_071024.shp

Add --reaches to also write a reach-level dissolve alongside or
1.4_true_reach_polygons.py does the same thing with reach-level
attributes attached.

--exclude-reaches drops named SWORD reaches from the output:

    --exclude-reaches 81270500291
    --exclude-reaches exclude_YR_AOI.txt

(But this is a POST-HOC filter and does not re-run anything.)

Requires: rasterio, geopandas, shapely, numpy, netCDF4, pandas

-----------------------------------------------------------------------------
Script by:
Camryn Kluetmeier (camryn.kluetmeier@duke.edu)

Parts of this script were developed with assistance from Claude Code
(Anthropic) for debugging, documentation, and related editorial suggestions.

Last updated: 2026-09-15

"""

PIPELINE_VERSION = '1.2.0'

import argparse
import os
import sys

import numpy as np
import netCDF4
import rasterio
from rasterio import features
import geopandas as gpd
from shapely.geometry import shape as shapely_shape
from shapely.ops import unary_union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reach_exclusions


def build(water_tif, pixcvec_nc, key='node_id', exclude=()):
    """
    Dissolve the water mask by whichever RiverObs id `key` names.

    Returns (GeoDataFrame, n_assigned_cells, n_excluded_cells). The two
    counts are reported separately so report_coverage() can tell water that
    no reach ever claimed apart from water deliberately dropped here.
    """
    with netCDF4.Dataset(pixcvec_nc, 'r') as ds:
        col = np.asarray(ds['range_index'][:]).astype(np.int64)
        row = np.asarray(ds['azimuth_index'][:]).astype(np.int64)
        ids = np.asarray(ds[key][:]).astype(np.int64)
        reach = (ids if key == 'reach_id'
                 else np.asarray(ds['reach_id'][:]).astype(np.int64))

    n_excluded = 0
    if exclude:
        keep = ~np.isin(reach, np.asarray(sorted(exclude), dtype=np.int64))
        n_excluded = int((~keep).sum())
        present = sorted(set(int(r) for r in np.unique(reach)) & set(exclude))
        absent = sorted(set(exclude) - set(int(r) for r in np.unique(reach)))
        print('  excluding {} of {} assigned cells, from reach(es) {}'.format(
            n_excluded, len(ids), ', '.join(str(r) for r in present) or 'none'))
        if absent:
            print('  note: excluded reach(es) {} are not in this PIXCVec at '
                  'all -- they had no effect here. If step 1.2 already '
                  'dropped them from the prior database, that is expected.'
                  .format(', '.join(str(r) for r in absent)))
        if not n_excluded:
            print('  note: nothing was excluded -- the output is identical '
                  'to an unfiltered run')
        col, row, ids = col[keep], row[keep], ids[keep]
        if not len(ids):
            raise SystemExit(
                'every assigned cell was excluded -- nothing left to '
                'polygonize. Check the exclusion list against the reach_id '
                'values in the RiverTile.')

    with rasterio.open(water_tif) as src:
        shape, transform, crs = (src.height, src.width), src.transform, src.crs

    if row.max() >= shape[0] or col.max() >= shape[1]:
        raise SystemExit(
            'PIXCVec indices exceed the raster shape -- the water .tif and the '
            'PIXCVec file are from different runs. They must be the same pair.')

    # rasterio.features.shapes cannot polygonize int64, so map the ids
    # to a dense int32 label and map back afterwards.
    uniq, inverse = np.unique(ids, return_inverse=True)
    label = np.zeros(shape, dtype=np.int32)          # 0 = unassigned
    label[row, col] = inverse.astype(np.int32) + 1

    print('{} water cells assigned to {} distinct {}s'.format(
        len(ids), len(uniq), key))

    geoms, values = [], []
    for geom, val in features.shapes(label, mask=label > 0, transform=transform):
        geoms.append(shapely_shape(geom))
        values.append(int(val))
    values = np.asarray(values)

    out_geom, out_id, out_area = [], [], []
    for v in np.unique(values):
        parts = [g for g, vv in zip(geoms, values) if vv == v]
        merged = unary_union(parts)
        out_geom.append(merged)
        out_id.append(int(uniq[v - 1]))
        out_area.append(merged.area)

    counts = {int(uniq[v - 1]): int((inverse == (v - 1)).sum())
              for v in np.unique(values)}

    gdf = gpd.GeoDataFrame({
        key: out_id,
        # String copy: shapefile DBF numeric fields are the usual place
        # a 14-digit id gets mangled. Round-trips fine in testing, but
        # if the numeric column ever looks wrong, this one is
        # authoritative.
        key.replace('_id', '_str'): [str(i) for i in out_id],
        'n_cells': [counts.get(i, 0) for i in out_id],
        'area_m2': out_area,
        # Edited in QGIS. 1 = keep, 0 = drop. Read by the manual node QC
        # step (3.1.4_manual_node_qc.py, which lives with the width-
        # validation scripts, not in this folder).
        'keep': [1] * len(out_id),
    }, geometry=out_geom, crs=crs)
    n_assigned = int((label > 0).sum())
    return (gdf.sort_values(key).reset_index(drop=True), n_assigned,
            n_excluded)


def report_provenance(pixcvec_nc):
    """What corridor and what exclusion list produced this PIXCVec?"""
    with netCDF4.Dataset(pixcvec_nc, 'r') as ds:
        wth = getattr(ds, 'riverobs_wth_coef_factor', None)
        ext = getattr(ds, 'riverobs_ext_dist_coef_factor', None)
        excl = getattr(ds, 'riverobs_excluded_reaches', None)
        excl_missing = getattr(
            ds, 'riverobs_excluded_reaches_not_in_tile', None)

    if wth is None and ext is None:
        print('  corridor factors: not recorded on this PIXCVec (written before '
              'run_calval2rivertile 1.1.0, or by calval2rivertile directly)')
    else:
        stock = (wth in (None, 1.0)) and (ext in (None, 1.0))
        print('  corridor factors: wth_coef x{}, ext_dist_coef x{}{}'.format(
            wth, ext, '  (stock RiverObs)' if stock else ''))

    if excl is None:
        print('  prior-database exclusions: not recorded (written before '
              'run_calval2rivertile 1.2.0)')
    elif not str(excl).strip():
        print('  prior-database exclusions: none -- every SWORD reach in the '
              'bounding box competed for water')
    else:
        print('  prior-database exclusions: {} were removed BEFORE assignment, '
              'so the kept reaches here may have absorbed their water'.format(
                  excl))
        if excl_missing and str(excl_missing).strip():
            print('    (asked for but not in the tile, so ignored: {})'.format(
                excl_missing))


def report_coverage(water_tif, n_assigned_cells, n_excluded_cells=0):
    """How much digitized water reached a node, and how much we dropped."""
    with rasterio.open(water_tif) as src:
        n_water = int((src.read(1) == 1).sum())
        cell_area = abs(src.transform[0] * src.transform[4])
    if not n_water:
        return
    pct = 100.0 * n_assigned_cells / n_water
    print('  mask cells assigned to a node: {:,} of {:,}  ({:.2f}%)'.format(
        n_assigned_cells, n_water, pct))
    if n_excluded_cells:
        print('  of which {:,} ({:,.0f} m2) were dropped by --exclude-reaches '
              'and are not in the output'.format(
                  n_excluded_cells, n_excluded_cells * cell_area))
    if pct < 99.0:
        unclaimed = n_water - n_assigned_cells
        print('  {:,.0f} m2 of digitized water never reached a node.'.format(
            unclaimed * cell_area))
        if n_excluded_cells:
            # Do not send someone to widen the corridor when the gap is
            # water they asked to drop, or water whose reach step 1.2
            # removed from the prior database.
            print('  Some of that is expected here: reaches were excluded. '
                  'Check the count above,')
            print('  and the prior-database exclusions line, before changing '
                  'the corridor.')
        else:
            print('  If that water is real river, the corridor is too tight '
                  '-- re-run step 2 with')
            print('  larger --wth-coef-factor / --ext-dist-coef-factor, then '
                  're-run this script on')
            print('  the new pixcvec.')


def join_rivertile(gdf, rivertile_nc):
    """
    Attach width, p_length and p_dist_out from the RiverTile so the QGIS layer
    is directly judgeable: style by `width` and a node truncated by cloud or by
    the survey edge stands out as anomalously narrow against its neighbours.

    p_dist_out is also what orders nodes along stream, so it is the x axis of
    any downstream profile.
    """
    import pandas as pd
    with netCDF4.Dataset(rivertile_nc, 'r') as ds:
        n = ds.groups['nodes']
        get = lambda k: np.ma.filled(n[k][:].astype('f8'), np.nan)
        rt = pd.DataFrame({
            'node_id': get('node_id').astype('int64'),
            'reach_id': get('reach_id').astype('int64'),
            'width': get('width'),
            'p_length': get('p_length'),
            'p_dist_out': get('p_dist_out'),
            'n_good_pix': get('n_good_pix'),
        })
    merged = gdf.merge(rt, on='node_id', how='left')
    return gpd.GeoDataFrame(merged, geometry=gdf.geometry.values, crs=gdf.crs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('water_tif',
                   help='the water raster from '
                        '1.1_rasterize_watermask.py')
    p.add_argument('pixcvec_nc', help='the *_pixcvec.nc from the same run')
    p.add_argument('out_shp')
    p.add_argument('--rivertile', default=None,
                   help='the RiverTile .nc from the same run. Recommended: '
                        'adds reach_id / width / p_length / p_dist_out so you '
                        'can style the layer by width during manual review '
                        'and order it along stream.')
    p.add_argument('--reaches', action='store_true',
                   help='also write a bare reach-level dissolve alongside. '
                        '1.4_true_reach_polygons.py does this with reach '
                        'attributes attached.')
    reach_exclusions.add_argument(p)
    # Accepted only so the mistake gets an explanation instead of
    # argparse's 'unrecognized arguments'. The corridor is set in step
    # 2; see the header.
    p.add_argument('--wth-coef-factor', '--ext-dist-coef-factor',
                   dest='corridor_attempt', default=None,
                   help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.corridor_attempt is not None:
        raise SystemExit(
            'The search corridor is not set here.\n'
            'This script does not run RiverObs -- it reads the node_id that '
            'RiverObs already\nwrote into the PIXCVec and dissolves the mask by '
            'it. The corridor decided which\ncells got a node_id at all, back in '
            'step 2, and that is frozen in the file.\n\n'
            'Set the factors where the assignment happens, then re-run this '
            'script on the\nNEW pixcvec:\n\n'
            '    python run_calval2rivertile.py <water.tif> airborne_watermask '
            '\\\n'
            '        <rivertile.nc> <pixcvec.nc> <rdf> <pixc.nc> \\\n'
            '        --riverobs-root /path/to/RiverObs \\\n'
            '        --wth-coef-factor 3.0 --ext-dist-coef-factor 3.0\n\n'
            'or, for the survey batch:  1.2_run_RiverObs.py '
            '--wth-coef-factor 3.0 ...')
    print('# {} {}'.format(os.path.basename(__file__), PIPELINE_VERSION))

    try:
        excluded = reach_exclusions.parse(args.exclude_reaches)
    except reach_exclusions.ExclusionError as exc:
        raise SystemExit('--exclude-reaches: {}'.format(exc))
    if excluded:
        print('post-hoc exclusion of {} (assignment is not re-done)'.format(
            reach_exclusions.describe(excluded, limit=1000)))

    gdf, n_assigned, n_excluded = build(
        args.water_tif, args.pixcvec_nc, 'node_id', excluded)
    if args.rivertile:
        gdf = join_rivertile(gdf, args.rivertile)
    gdf.to_file(args.out_shp)
    print('wrote {} node polygons to {}'.format(len(gdf), args.out_shp))
    print('  total assigned area: {:,.0f} m2'.format(gdf.area_m2.sum()))
    print('  per-node area: min {:,.0f}  median {:,.0f}  max {:,.0f} m2'.format(
        gdf.area_m2.min(), gdf.area_m2.median(), gdf.area_m2.max()))
    report_provenance(args.pixcvec_nc)
    report_coverage(args.water_tif, n_assigned, n_excluded)

    if args.reaches:
        rgdf, _, _ = build(
            args.water_tif, args.pixcvec_nc, 'reach_id', excluded)
        rpath = args.out_shp.replace('.shp', '_reaches.shp')
        rgdf.to_file(rpath)
        print('wrote {} reach polygons to {}'.format(len(rgdf), rpath))
        print('  (for reach attributes and along-stream ordering, use '
              '1.4_true_reach_polygons.py)')

    if 'width' in gdf.columns:
        w = gdf['width'].dropna()
        if len(w):
            print('  width: min {:.1f}  median {:.1f}  max {:.1f} m'.format(
                w.min(), w.median(), w.max()))
            narrow = gdf[gdf['width'] < 0.6 * w.median()]
            if len(narrow):
                print('  {} node(s) narrower than 60% of the median -- worth a '
                      'look first in QGIS:'.format(len(narrow)))
                for _, r in narrow.sort_values('width').head(10).iterrows():
                    print('     {}  width {:.1f} m'.format(
                        int(r['node_id']), r['width']))

    print('\nNEXT: open in QGIS over the water mask, set `keep` = 0 on nodes to')
    print('drop (cloud, survey edge, a channel you do not want), save, then')
    print('build the reach layer with 1.4_true_reach_polygons.py.')


if __name__ == '__main__':
    main()

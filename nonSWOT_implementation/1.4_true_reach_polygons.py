#!/usr/bin/env python
"""Build reach footprints from RiverObs's pixel assignment.

Dissolves the water mask by the reach_id RiverObs wrote into the PIXCVec.

Usage:
    python 1.4_true_reach_polygons.py \\
        /path/rasters/YR_AOI_water_10m.tif \\
        /path/riverobs_out/v17b/ortho_YR_AOI_v17b_pixcvec.nc \\
        /path/out/true_reach_polygons_YR_AOI.shp \\
        --rivertile /path/riverobs_out/v17b/ortho_YR_AOI_v17b.nc

`--rivertile` attaches the reach-level RiverTile fields (e.g. p_dist_out)


EXCLUDING REACHES
-----------------
    --exclude-reaches 81270500291
    --exclude-reaches exclude_YR_AOI.txt

This is a POST-HOC filter on a finished PIXCVec and does not re-run
anything.

Requires: rasterio, geopandas, shapely, numpy, netCDF4, pandas

-----------------------------------------------------------------------------
Script by:
Camryn Kluetmeier (camryn.kluetmeier@duke.edu)

Parts of this script were developed with assistance from Claude Code
(Anthropic) for debugging, documentation, and related editorial suggestions.

Last updated: 2026-09-15

"""

PIPELINE_VERSION = '1.0.0'

import argparse
import importlib.util
import os
import sys

import numpy as np
import netCDF4
import geopandas as gpd
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reach_exclusions


# Reach-level RiverTile fields attached to the polygons. Every name is 10
# characters or fewer, because a shapefile DBF truncates longer field
# names and two fields that collide after truncation silently overwrite
# each other.
#
# `width` here is sum(node area) / sum(node p_length) over the observed
# nodes -- a length-weighted mean, not the unweighted mean of node
# widths. `obs_frac_n` is the fraction of the reach's SWORD nodes that
# received data and is the defensible basis for a completeness
# threshold. wse and slope are deliberately absent: the pixel cloud has
# constant zero heights, so they carry no information in this pipeline.
REACH_FIELDS = [
    'width', 'width_u', 'area_total', 'area_detct',
    'obs_frac_n', 'partial_f', 'n_good_nod',
    'p_length', 'p_width', 'p_dist_out', 'p_n_nodes',
]


def _load_sibling(module_name, filename):
    """
    Import a module whose filename is not a valid Python identifier.

    1.3_true_node_polygons.py cannot be imported with `import` because the
    name starts with a digit, and its dissolve is exactly the code this
    script needs. Loading it by path keeps one implementation of the
    dissolve rather than two that can drift.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if not os.path.exists(path):
        raise SystemExit(
            '{} must sit next to this script; it holds the dissolve and the '
            'coverage report.'.format(filename))
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_nodes = _load_sibling('true_node_polygons', '1.3_true_node_polygons.py')


def count_observed_nodes(pixcvec_nc, exclude=()):
    """Distinct node_ids per reach in the PIXCVec.

    Not the same as the RiverTile's n_good_nod, which counts nodes passing
    its own quality gate. This is simply how many of the reach's nodes
    received at least one water cell, and it is the number to check when a
    reach's mean looks off.

    Counted by uniquing node_id alone and taking each unique node's reach,
    rather than uniquing (reach_id, node_id) rows. A PIXCVec over a 10 m
    mask runs to tens of millions of points, and np.unique(..., axis=0) on
    a two-column array of that length sorts a void view of the whole thing
    -- several times the memory of the 1-D sort, for no extra information:
    a SWORD node_id already encodes its reach, and RiverObs writes one
    node per pixel, so node_id -> reach_id is a function here.
    """
    with netCDF4.Dataset(pixcvec_nc, 'r') as ds:
        reach = np.asarray(ds['reach_id'][:]).astype(np.int64)
        node = np.asarray(ds['node_id'][:]).astype(np.int64)
    if exclude:
        keep = ~np.isin(reach, np.asarray(sorted(exclude), dtype=np.int64))
        reach, node = reach[keep], node[keep]
    _, first = np.unique(node, return_index=True)
    ids, counts = np.unique(reach[first], return_counts=True)
    return dict(zip(ids.tolist(), counts.tolist()))


def join_rivertile(gdf, rivertile_nc):
    """Attach the reach-level RiverTile fields, plus river_name."""
    with netCDF4.Dataset(rivertile_nc, 'r') as ds:
        grp = ds.groups['reaches']
        data = {'reach_id': np.ma.filled(
            grp['reach_id'][:], -1).astype('int64').ravel()}
        for field in REACH_FIELDS:
            if field not in grp.variables:
                print('  note: {} not in /reaches -- skipping'.format(field))
                continue
            data[field] = np.ma.filled(
                grp[field][:].astype('f8'), np.nan).ravel()
        if 'river_name' in grp.variables:
            data['river_name'] = [
                str(v) for v in np.asarray(grp['river_name'][:]).ravel()]
    rt = pd.DataFrame(data)
    merged = gdf.merge(rt, on='reach_id', how='left')
    missing = int(merged['p_dist_out'].isna().sum()) if \
        'p_dist_out' in merged.columns else 0
    if missing:
        print('  note: {} polygon(s) found no matching reach in the '
              'RiverTile. The .tif, the PIXCVec and the RiverTile must all '
              'come from the same run.'.format(missing))
    return gpd.GeoDataFrame(merged, geometry=gdf.geometry.values, crs=gdf.crs)


def report_river_mix(gdf):
    """
    Warn when the polygons span more than one river.

    ds_rank orders by p_dist_out, the SWORD distance to the OUTLET. Two
    rivers in the same basin are measured to the same outlet, so their
    p_dist_out ranges overlap while the reaches sit on different channels.
    Ordering the whole layer by ds_rank then interleaves them, and a
    profile drawn from it steps sideways onto a tributary and back without
    anything in the plot saying so.

    p_dist_out is a valid along-stream axis only along ONE continuous path
    through the network. Pick the river first.
    """
    named = gdf[['river_name', 'p_dist_out']].copy() \
        if 'p_dist_out' in gdf.columns else None
    if named is None:
        return
    groups = []
    for name, sub in named.groupby('river_name'):
        lo, hi = sub['p_dist_out'].min(), sub['p_dist_out'].max()
        groups.append((name, len(sub), lo, hi))
    if len(groups) <= 1:
        return
    print('\n  These polygons span {} river_name values:'.format(len(groups)))
    for name, n, lo, hi in sorted(groups, key=lambda g: -g[1]):
        print('     {:<20s} {:3d} reach(es), p_dist_out {:,.0f}-{:,.0f} km'
              .format(str(name), n, lo / 1000.0, hi / 1000.0))
    overlapping = [
        (a, b) for i, a in enumerate(groups) for b in groups[i + 1:]
        if a[2] <= b[3] and b[2] <= a[3]]
    if overlapping:
        print('  Their p_dist_out ranges OVERLAP, so ds_rank interleaves them '
              'and a single profile')
        print('  drawn across the whole layer would step off one river onto '
              'another. p_dist_out is')
        print('  an along-stream axis only along one continuous path -- filter '
              'to one river')
        print('  (river_name, or --exclude-reaches) before plotting.')
    else:
        print('  Their p_dist_out ranges do not overlap, but they are still '
              'different channels --')
        print('  filter to one river before reading ds_rank as a profile '
              'axis.')


def report_bifurcations(rivertile_nc, kept_ids, excluded=()):
    """
    List every reach where SWORD branches, and name the branches.

    A reach with n_reach_up > 1 (or n_reach_dn > 1) is where the prior
    network splits. Its branch ids are the candidates for exclusion.
    Reported straight from the product's own topology variables rather
    than inferred from geometry.

    Not every branch is a parallel channel, and the difference decides
    whether excluding one is right at all, so each candidate is
    classified by two signals the product already carries:

      * a branch that appears at TWO different branching reaches leaves
        the network at one and rejoins it at the other. That is a side
        channel around a stem -- exactly the case a single-channel
        profile wants gone.
      * a branch that appears once is a separate arm. If SWORD gives it
        the same river_name as the reach it branches from, it is most
        likely a split of that river; a different name means a tributary
        or distributary, which a profile of the stem should simply not
        follow rather than exclude.

    Both are heuristics over SWORD's own labelling, not a judgement about
    the river. Look at the pair in QGIS before excluding either.
    """
    with netCDF4.Dataset(rivertile_nc, 'r') as ds:
        grp = ds.groups['reaches']
        needed = ('reach_id', 'n_reach_up', 'n_reach_dn',
                  'rch_id_up', 'rch_id_dn')
        if any(k not in grp.variables for k in needed):
            print('  (no bifurcation report: this RiverTile is missing one of '
                  '{})'.format(', '.join(needed)))
            return
        rid = np.ma.filled(grp['reach_id'][:], -1).astype('int64').ravel()
        n_up = np.ma.filled(grp['n_reach_up'][:], 0).astype(int).ravel()
        n_dn = np.ma.filled(grp['n_reach_dn'][:], 0).astype(int).ravel()
        up = np.atleast_2d(np.ma.filled(grp['rch_id_up'][:], 0).astype('int64'))
        dn = np.atleast_2d(np.ma.filled(grp['rch_id_dn'][:], 0).astype('int64'))
        names = {}
        if 'river_name' in grp.variables:
            names = {int(k): str(v) for k, v in zip(
                rid, np.asarray(grp['river_name'][:]).ravel())}

    branching = np.where((n_up > 1) | (n_dn > 1))[0]
    print('\nSWORD bifurcations in this tile: {} of {} reaches'.format(
        len(branching), len(rid)))
    if not len(branching):
        print('  none -- every reach has at most one upstream and one '
              'downstream neighbour, so there are no parallel channels to '
              'exclude.')
        return

    kept = set(int(i) for i in kept_ids)
    excluded = set(int(i) for i in excluded)
    appears = {}          # branch id -> [(parent reach id, side), ...]
    for i in branching:
        for side, count, arr in (('up', n_up[i], up[i]), ('dn', n_dn[i], dn[i])):
            if count <= 1:
                continue
            branches = [int(v) for v in arr if v > 0]
            marks = []
            for b in branches:
                if b in excluded:
                    marks.append('{} [excluded]'.format(b))
                elif b in kept:
                    marks.append('{} [in output]'.format(b))
                else:
                    marks.append('{} [not observed]'.format(b))
                siblings = tuple(o for o in branches if o != b)
                appears.setdefault(b, []).append((int(rid[i]), side, siblings))
            print('  reach {} has {} {}stream branches: {}'.format(
                rid[i], count, side, ', '.join(marks)))

    def _named(rid_):
        """river_name, or None when SWORD does not record one."""
        name = names.get(rid_)
        if not name or name in ('no_data', 'NODATA', '--'):
            return None
        return name

    def classify(b):
        """(rank, one-line reason). Lower rank = stronger exclusion candidate."""
        where = appears.get(b, [])
        if len(where) > 1:
            return 0, ('side channel -- branches off {} and rejoins at {}'
                        .format(where[0][0], where[1][0]))
        if not where:
            return 3, 'no branching record'
        parent, _side, siblings = where[0]
        opposite = (', opposite {}'.format(
            ' / '.join(str(s) for s in siblings)) if siblings else '')
        mine, theirs = _named(b), _named(parent)
        if mine and theirs and mine == theirs:
            return 1, ('one arm of the split at {}{} -- same river_name ({})'
                       .format(parent, opposite, mine))
        if mine and theirs:
            return 2, ('branches off {}{} -- different river_name ({} vs {}), '
                       'so more likely a tributary or distributary than a '
                       'parallel channel'.format(parent, opposite, mine, theirs))
        return 2, ('branches off {}{} -- river_name not recorded for both, so '
                   'no name evidence either way'.format(parent, opposite))

    still = sorted(b for b in appears if b in kept and b not in excluded)
    if not still:
        print('\n  No branch reach is still in the output.')
        return appears
    ranked = sorted(still, key=lambda b: (classify(b)[0], b))
    print('\n  {} branch reach(es) are still in the output:'.format(len(ranked)))
    for b in ranked:
        print('    {}  {}'.format(b, classify(b)[1]))

    side_channels = [b for b in ranked if classify(b)[0] == 0]
    if side_channels:
        print('\n  The side channel(s) are the clear candidates for a '
              'single-channel profile:')
        print('    --exclude-reaches ' + ' '.join(str(b)
                                                  for b in side_channels))
    else:
        print('\n  None of them leaves and rejoins the network, so none is an '
              'unambiguous side')
        print('  channel. Exclude one only if you have decided it is not the '
              'stem you are')
        print('  profiling -- these are more likely confluences and '
              'tributaries.')
    print('  Look at the pair in QGIS first: which branch is the stem is a '
          'judgement about')
    print('  the river, not something this script can decide.')
    return appears


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('water_tif',
                   help='the water raster from 1.1_rasterize_watermask.py')
    p.add_argument('pixcvec_nc', help='the *_pixcvec.nc from the same run')
    p.add_argument('out_shp')
    p.add_argument('--rivertile', default=None,
                   help='the RiverTile .nc from the same run. Strongly '
                        'recommended: adds the reach-level fields including '
                        'p_dist_out, and enables the bifurcation report.')
    reach_exclusions.add_argument(p)
    p.add_argument('--min-obs-frac', type=float, default=None,
                   metavar='FRAC',
                   help='also set keep = 0 on reaches whose obs_frac_n is '
                        'below FRAC (e.g. 0.5). Needs --rivertile. Nothing '
                        'is deleted -- the polygons stay in the file so you '
                        'can see what was flagged.')
    p.add_argument('--wth-coef-factor', '--ext-dist-coef-factor',
                   dest='corridor_attempt', default=None,
                   help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.corridor_attempt is not None:
        raise SystemExit(
            'The search corridor is not set here.\n'
            'This script does not run RiverObs -- it reads the reach_id that '
            'RiverObs already\nwrote into the PIXCVec and dissolves the mask '
            'by it. Set the factors in step 2\n(1.2_run_RiverObs.py or '
            'run_calval2rivertile.py), then re-run this script on the\nnew '
            'pixcvec.')
    print('# {} {}'.format(os.path.basename(__file__), PIPELINE_VERSION))

    if args.min_obs_frac is not None and not args.rivertile:
        raise SystemExit('--min-obs-frac needs --rivertile: obs_frac_n comes '
                         'from the RiverTile, not the PIXCVec.')

    try:
        excluded = reach_exclusions.parse(args.exclude_reaches)
    except reach_exclusions.ExclusionError as exc:
        raise SystemExit('--exclude-reaches: {}'.format(exc))
    if excluded:
        print('post-hoc exclusion of {} (assignment is not re-done)'.format(
            reach_exclusions.describe(excluded, limit=1000)))

    gdf, n_assigned, n_excluded = _nodes.build(
        args.water_tif, args.pixcvec_nc, 'reach_id', excluded)

    obs_nodes = count_observed_nodes(args.pixcvec_nc, excluded)
    gdf['n_nodes_ob'] = [obs_nodes.get(int(r), 0) for r in gdf['reach_id']]
    gdf['n_parts'] = [
        1 if g.geom_type == 'Polygon' else len(g.geoms) for g in gdf.geometry]

    if args.rivertile:
        gdf = join_rivertile(gdf, args.rivertile)
        if 'p_dist_out' in gdf.columns:
            # 1 = furthest downstream observed reach. p_dist_out is the SWORD
            # distance to the outlet, so ascending p_dist_out is downstream
            # to upstream. NaNs are ranked last rather than dropped.
            gdf['ds_rank'] = (gdf['p_dist_out']
                              .rank(method='first', na_option='bottom')
                              .astype('int64'))
            gdf = gdf.sort_values('ds_rank').reset_index(drop=True)

    if args.min_obs_frac is not None and 'obs_frac_n' in gdf.columns:
        low = gdf['obs_frac_n'] < args.min_obs_frac
        gdf.loc[low, 'keep'] = 0
        print('  flagged keep = 0 on {} reach(es) with obs_frac_n < '
              '{}'.format(int(low.sum()), args.min_obs_frac))

    gdf.to_file(args.out_shp)
    print('wrote {} reach polygons to {}'.format(len(gdf), args.out_shp))
    print('  total assigned area: {:,.0f} m2'.format(gdf.area_m2.sum()))
    print('  per-reach area: min {:,.0f}  median {:,.0f}  max {:,.0f} m2'
          .format(gdf.area_m2.min(), gdf.area_m2.median(), gdf.area_m2.max()))

    multi = gdf[gdf['n_parts'] > 1]
    if len(multi):
        print('  {} reach(es) are multipart (water split around bars or a '
              'gap in the mask):'.format(len(multi)))
        for _, r in multi.sort_values('n_parts', ascending=False).head(5).iterrows():
            print('     {}  {} parts'.format(int(r['reach_id']),
                                             int(r['n_parts'])))

    _nodes.report_provenance(args.pixcvec_nc)
    _nodes.report_coverage(args.water_tif, n_assigned, n_excluded)

    if 'obs_frac_n' in gdf.columns:
        partial = gdf[gdf['obs_frac_n'] < 0.5]
        if len(partial):
            print('  {} reach(es) observed on under half their SWORD nodes '
                  '-- their reach means are not comparable to a fully '
                  'observed reach:'.format(len(partial)))
            for _, r in partial.sort_values('obs_frac_n').head(10).iterrows():
                print('     {}  obs_frac_n {:.2f}  ({} of {:.0f} nodes '
                      'observed)'.format(
                          int(r['reach_id']), r['obs_frac_n'],
                          int(r['n_nodes_ob']),
                          r.get('p_n_nodes', float('nan'))))

    if 'river_name' in gdf.columns:
        report_river_mix(gdf)

    if args.rivertile:
        report_bifurcations(args.rivertile, gdf['reach_id'].tolist(), excluded)

    print('\nNEXT: open in QGIS over the water mask. Order by ds_rank for an '
          'along-stream')
    print('profile; use the polygons as the zonal-statistics units for the '
          'Sentinel-2 and')
    print('SWOT raster variables.')


if __name__ == '__main__':
    main()

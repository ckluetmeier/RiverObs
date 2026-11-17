#!/usr/bin/env python
"""Build "flexible" reach footprints of a user-chosen length.

Groups consecutive SWORD nodes within each SWORD reach into sections of
roughly --length-km, then dissolves the node footprints of each group.

Usage:
    python 1.5_flexible_reach_polygons.py \\
        /path/rasters/YR_AOI_water_10m.tif \\
        /path/riverobs_out/v17b/ortho_YR_AOI_v17b_pixcvec.nc \\
        /path/out/flex_reach_polygons_YR_AOI_5km.shp \\
        --rivertile /path/riverobs_out/v17b/ortho_YR_AOI_v17b.nc \\
        --length-km 5

HOW A REACH IS SPLIT
--------------------
For each SWORD reach the script takes all of its prior nodes from the
RiverTile /nodes group (observed or not), orders them downstream ->
upstream by p_dist_out, and sums their p_length to get the prior reach
length T. It then picks the number of sections k (1 <= k <= n_nodes)
whose length T/k is closest to the requested length, and cuts the node
sequence into k contiguous groups of near-equal prior length (a node goes
to the group containing its along-stream midpoint).

Because k is a whole number and reaches are ~7-15 km, the actual
lengths vary a bit.

ATTRIBUTES
----------------------------------------------------------
  flex_id / flex_str   section id (number / text copy)
  flex_num, n_flex     section number within the reach, sections in reach
  reach_id / reach_str original SWORD reach
  node_ids             nodes merged into the section
  node_dn, node_up     most downstream / upstream node in the section
  n_nodes_p            prior SWORD nodes in the section
  n_nodes_ob           of those, nodes that received water
  obs_frac_n           n_nodes_ob / n_nodes_p
  flex_len             prior centreline length, m (sum of node p_length)
  target_len           the requested length, m
  width                sum(node width * p_length) / sum(p_length) over
                       observed nodes, i.e. node area / length -- the same
                       definition as the reach width in 1.4
  wid_poly             polygon area_m2 / flex_len (includes every cell
                       assigned, unobserved length counted in the divisor)
  n_good_pix           sum of node n_good_pix
  p_dist_out           length-weighted mean node p_dist_out, m (section
                       midpoint); dist_dn / dist_up are its two ends
  ds_rank              1 = most downstream section (by p_dist_out)
  r_*                  the original reach's RiverTile fields from 1.4,
                       renamed with an r_ prefix

EXCLUDING REACHES
-----------------
    --exclude-reaches 81270500291
    --exclude-reaches exclude_YR_AOI.txt

-----------------------------------------------------------------------------
Script by:
Camryn Kluetmeier (camryn.kluetmeier@duke.edu)

Parts of this script were developed with assistance from Claude
(Anthropic) for code drafting, debugging and documentation.

Last updated: 2026-11-17

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
from shapely.ops import unary_union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reach_exclusions

DBF_TEXT_MAX = 254

# Reach-level RiverTile fields (from 1.4) renamed so they cannot collide
# with the section-level fields of the same name. All <= 10 characters.
REACH_RENAME = {
    'width': 'r_width', 'width_u': 'r_width_u',
    'area_total': 'r_area_tot', 'area_detct': 'r_area_det',
    'obs_frac_n': 'r_obs_frac', 'partial_f': 'r_partial',
    'n_good_nod': 'r_n_good', 'p_length': 'r_p_length',
    'p_width': 'r_p_width', 'p_dist_out': 'r_p_dist',
    'p_n_nodes': 'r_n_nodes',
}


def _load_sibling(module_name, filename):
    """Import a pipeline step whose filename starts with a digit."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if not os.path.exists(path):
        raise SystemExit('{} must sit next to this script.'.format(filename))
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_nodes = _load_sibling('true_node_polygons', '1.3_true_node_polygons.py')
_reaches = _load_sibling('true_reach_polygons', '1.4_true_reach_polygons.py')


# --------------------------------------------------------------------------
# SWORD id helpers. node_id = CBBBBBRRRR NNN T, reach_id = CBBBBBRRRR T.
# --------------------------------------------------------------------------
def node_number(node_id):
    return (np.asarray(node_id, dtype=np.int64) // 10) % 1000


def make_flex_id(reach_id, flex_num):
    reach_id = np.asarray(reach_id, dtype=np.int64)
    return (reach_id // 10) * 10000 + np.asarray(flex_num, np.int64) * 10 \
        + reach_id % 10


# --------------------------------------------------------------------------
def read_rivertile_nodes(rivertile_nc):
    """Every prior node the RiverTile lists, observed or not."""
    with netCDF4.Dataset(rivertile_nc, 'r') as ds:
        n = ds.groups['nodes']

        def get(k, fill=np.nan, dtype='f8'):
            if k not in n.variables:
                return None
            return np.ma.filled(n[k][:].astype(dtype), fill).ravel()

        out = {
            'node_id': get('node_id', -1, 'i8'),
            'reach_id': get('reach_id', -1, 'i8'),
        }
        for k in ('p_length', 'p_dist_out', 'width', 'n_good_pix'):
            v = get(k)
            out[k] = v if v is not None else np.full(len(out['node_id']),
                                                     np.nan)
            if v is None:
                print('  note: {} not in /nodes -- left empty'.format(k))
    df = pd.DataFrame(out)
    df = df[(df.node_id > 0) & (df.reach_id > 0)]
    # Fill values in the product are huge negatives / masked -> NaN above;
    # guard the integer-ish count against the int fill too.
    df.loc[df.n_good_pix < 0, 'n_good_pix'] = np.nan
    return df.drop_duplicates('node_id').reset_index(drop=True)


def choose_k(total_len, n_nodes, target):
    """Number of sections whose length T/k is closest to target."""
    if not np.isfinite(total_len) or total_len <= 0 or n_nodes <= 1:
        return 1
    ks = np.arange(1, n_nodes + 1)
    err = np.abs(total_len / ks - target)
    return int(ks[np.argmin(err)])        # argmin -> ties go to fewer sections


def partition(lengths, target):
    """
    Split an ordered node sequence into contiguous groups of near-equal
    length. Returns 1-based group numbers (consecutive, downstream first).
    """
    lengths = np.asarray(lengths, dtype='f8')
    total = lengths.sum()
    k = choose_k(total, len(lengths), target)
    if k == 1:
        return np.ones(len(lengths), dtype=np.int64)
    mid = np.cumsum(lengths) - lengths / 2.0
    grp = np.clip(np.floor(mid / (total / k)).astype(np.int64), 0, k - 1)
    # A very long node can leave a group empty; renumber consecutively.
    _, grp = np.unique(grp, return_inverse=True)
    return grp.astype(np.int64) + 1


def assign_sections(node_tab, target_m):
    """
    Add flex_num / n_flex / flex_id / order to node_tab (one row per
    prior node of every reach that is in the output).
    """
    pieces, n_order_mismatch, n_by_number = [], 0, 0
    for rid, sub in node_tab.groupby('reach_id', sort=True):
        sub = sub.copy()
        sub['nnum'] = node_number(sub['node_id'])
        if sub['p_dist_out'].notna().all():
            sub = sub.sort_values(['p_dist_out', 'nnum'])
            if not sub['nnum'].is_monotonic_increasing:
                n_order_mismatch += 1
        else:
            # Without p_dist_out, SWORD numbering (001 = downstream) is the
            # fallback order.
            sub = sub.sort_values('nnum')
            n_by_number += 1
        sub['flex_num'] = partition(sub['p_length'].values, target_m)
        sub['n_flex'] = int(sub['flex_num'].max())
        sub['order'] = np.arange(len(sub))
        pieces.append(sub)
    out = pd.concat(pieces, ignore_index=True)
    out['flex_id'] = make_flex_id(out['reach_id'], out['flex_num'])
    if n_order_mismatch:
        print('  note: in {} reach(es), node numbers do not increase with '
              'p_dist_out. Sections follow p_dist_out (the along-stream '
              'distance), not the id order.'.format(n_order_mismatch))
    if n_by_number:
        print('  note: {} reach(es) lack p_dist_out on some nodes and were '
              'ordered by SWORD node number instead.'.format(n_by_number))
    return out


def _ranges(nums):
    """[1,2,3,5,6] -> '001-003,005-006'."""
    nums = sorted(int(n) for n in nums)
    runs, start, prev = [], nums[0], nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        runs.append((start, prev))
        start = prev = n
    runs.append((start, prev))
    return ','.join('{:03d}'.format(a) if a == b
                    else '{:03d}-{:03d}'.format(a, b) for a, b in runs)


def summarise_sections(node_tab, target_m):
    """One attribute row per flexible reach."""
    rows = []
    for fid, s in node_tab.sort_values('order').groupby('flex_id', sort=False):
        obs = s[s['observed']]
        w_ok = obs[np.isfinite(obs['width']) & np.isfinite(obs['p_length'])]
        L = s['p_length']
        d_ok = s[np.isfinite(s['p_dist_out']) & np.isfinite(L)]
        rows.append({
            'flex_id': int(fid),
            'flex_str': str(int(fid)),
            'flex_num': int(s['flex_num'].iloc[0]),
            'n_flex': int(s['n_flex'].iloc[0]),
            'reach_id': int(s['reach_id'].iloc[0]),
            'reach_str': str(int(s['reach_id'].iloc[0])),
            '_ids_full': ','.join(str(int(i)) for i in s['node_id']),
            '_ids_rng': _ranges(s['nnum']),
            'node_dn': int(s['node_id'].iloc[0]),
            'node_up': int(s['node_id'].iloc[-1]),
            'n_nodes_p': int(len(s)),
            'n_nodes_ob': int(len(obs)),
            'obs_frac_n': len(obs) / float(len(s)),
            'flex_len': float(L.sum()),
            'target_len': float(target_m),
            'width': (float((w_ok['width'] * w_ok['p_length']).sum()
                            / w_ok['p_length'].sum())
                      if len(w_ok) and w_ok['p_length'].sum() > 0
                      else np.nan),
            'n_good_pix': (float(s['n_good_pix'].sum())
                           if s['n_good_pix'].notna().any() else np.nan),
            'p_dist_out': (float((d_ok['p_dist_out'] * d_ok['p_length']).sum()
                                 / d_ok['p_length'].sum())
                           if len(d_ok) and d_ok['p_length'].sum() > 0
                           else np.nan),
            'dist_dn': float(s['p_dist_out'].min()),
            'dist_up': float(s['p_dist_out'].max()),
        })
    return pd.DataFrame(rows)


def reach_attributes(reach_ids, rivertile_nc):
    """The 1.4 reach-level RiverTile fields, r_-prefixed."""
    stub = gpd.GeoDataFrame({'reach_id': sorted(set(reach_ids))},
                            geometry=[None] * len(set(reach_ids)))
    rt = pd.DataFrame(_reaches.join_rivertile(stub, rivertile_nc)
                      .drop(columns='geometry'))
    return rt.rename(columns=REACH_RENAME)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('water_tif',
                   help='the water raster from 1.1_rasterize_watermask.py')
    p.add_argument('pixcvec_nc', help='the *_pixcvec.nc from the same run')
    p.add_argument('out_shp', help='output .shp (or .gpkg to keep full '
                                   'node_id lists in the layer)')
    p.add_argument('--rivertile', required=True,
                   help='the RiverTile .nc from the same run. Required: the '
                        'prior node lengths and along-stream order come from '
                        'its /nodes group.')
    p.add_argument('--length-km', type=float, required=True, metavar='KM',
                   help='target flexible-reach length in km (e.g. 1, 2.5, 5)')
    reach_exclusions.add_argument(p)
    p.add_argument('--min-obs-frac', type=float, default=None, metavar='FRAC',
                   help='set keep = 0 on sections whose obs_frac_n is below '
                        'FRAC. Nothing is deleted.')
    args = p.parse_args()
    print('# {} {}'.format(os.path.basename(__file__), PIPELINE_VERSION))

    if not args.length_km > 0:
        raise SystemExit('--length-km must be > 0')
    target_m = args.length_km * 1000.0

    try:
        excluded = reach_exclusions.parse(args.exclude_reaches)
    except reach_exclusions.ExclusionError as exc:
        raise SystemExit('--exclude-reaches: {}'.format(exc))
    if excluded:
        print('post-hoc exclusion of {} (assignment is not re-done)'.format(
            reach_exclusions.describe(excluded, limit=1000)))

    # 1. Node footprints -- the exact dissolve 1.3 does.
    ngdf, n_assigned, n_excluded = _nodes.build(
        args.water_tif, args.pixcvec_nc, 'node_id', excluded)
    observed = set(int(i) for i in ngdf['node_id'])

    # 2. Prior node table for every reach that has water in the output.
    rt_nodes = read_rivertile_nodes(args.rivertile)
    node_reach = dict(zip(rt_nodes.node_id.tolist(), rt_nodes.reach_id.tolist()))
    missing = sorted(observed - set(node_reach))
    if missing:
        # Should not happen when the PIXCVec and RiverTile are one run.
        # Keep their water: attach to the reach their id encodes, with the
        # reach's median node length.
        print('  WARNING: {} observed node(s) are not in the RiverTile '
              '/nodes group (e.g. {}). Are the PIXCVec and RiverTile from '
              'the same run? Kept, ordered by node number.'.format(
                  len(missing), missing[0]))
        prefix_to_reach = {r // 10: r for r in rt_nodes.reach_id.unique()}
        extra = pd.DataFrame({
            'node_id': missing,
            'reach_id': [prefix_to_reach.get(n // 10000, -1) for n in missing],
            'p_length': np.nan, 'p_dist_out': np.nan,
            'width': np.nan, 'n_good_pix': np.nan})
        if (extra.reach_id < 0).any():
            raise SystemExit('observed nodes belong to a reach the RiverTile '
                             'does not list at all -- mismatched files.')
        rt_nodes = pd.concat([rt_nodes, extra], ignore_index=True)
        node_reach.update(zip(extra.node_id, extra.reach_id))
    kept_reaches = set(node_reach[n] for n in observed)
    node_tab = rt_nodes[rt_nodes.reach_id.isin(kept_reaches)].copy()
    node_tab['observed'] = node_tab.node_id.isin(observed)
    # Missing node lengths: reach median, else SWORD's nominal 200 m.
    med = node_tab.groupby('reach_id')['p_length'].transform('median')
    node_tab['p_length'] = node_tab['p_length'].fillna(med).fillna(200.0)

    # 3. Split each reach into sections.
    node_tab = assign_sections(node_tab, target_m)

    # 4. Dissolve node footprints into sections.
    n2f = dict(zip(node_tab.node_id, node_tab.flex_id))
    ngdf['flex_id'] = ngdf['node_id'].map(n2f).astype('int64')
    geoms, cells = {}, {}
    for fid, g in ngdf.groupby('flex_id'):
        geoms[fid] = unary_union(list(g.geometry))
        cells[fid] = int(g['n_cells'].sum())

    attrs = summarise_sections(node_tab, target_m)
    attrs = attrs[attrs.flex_id.isin(geoms)].reset_index(drop=True)
    attrs['n_cells'] = attrs.flex_id.map(cells)
    gdf = gpd.GeoDataFrame(attrs, geometry=[geoms[f] for f in attrs.flex_id],
                           crs=ngdf.crs)
    gdf['area_m2'] = gdf.geometry.area
    gdf['n_parts'] = [1 if g.geom_type == 'Polygon' else len(g.geoms)
                      for g in gdf.geometry]
    gdf['wid_poly'] = gdf['area_m2'] / gdf['flex_len']
    gdf['keep'] = 1

    # node_ids: full list if the format can hold it.
    is_shp = args.out_shp.lower().endswith('.shp')
    if is_shp and gdf['_ids_full'].str.len().max() > DBF_TEXT_MAX:
        gdf['node_ids'] = gdf['_ids_rng']
        print('  node_ids: full 14-digit lists exceed the 254-character '
              'shapefile limit, so\n  the layer stores node-number ranges '
              '(e.g. "001-025"). Full ids are in the\n  *_node_map.csv, or '
              'write .gpkg instead.')
    else:
        gdf['node_ids'] = gdf['_ids_full']
    gdf = gdf.drop(columns=['_ids_full', '_ids_rng'])

    # 5. Parent reach attributes.
    rattr = reach_attributes(gdf['reach_id'].tolist(), args.rivertile)
    gdf = gpd.GeoDataFrame(gdf.merge(rattr, on='reach_id', how='left'),
                           geometry='geometry', crs=ngdf.crs)

    gdf['ds_rank'] = (gdf['p_dist_out'].rank(method='first', na_option='bottom')
                      .astype('int64'))
    gdf = gdf.sort_values('ds_rank').reset_index(drop=True)

    if args.min_obs_frac is not None:
        low = gdf['obs_frac_n'] < args.min_obs_frac
        gdf.loc[low, 'keep'] = 0
        print('  flagged keep = 0 on {} section(s) with obs_frac_n < '
              '{}'.format(int(low.sum()), args.min_obs_frac))

    order = ['flex_id', 'flex_str', 'flex_num', 'n_flex', 'reach_id',
             'reach_str', 'node_ids', 'node_dn', 'node_up', 'n_nodes_p',
             'n_nodes_ob', 'obs_frac_n', 'flex_len', 'target_len', 'width',
             'wid_poly', 'n_good_pix', 'p_dist_out', 'dist_dn', 'dist_up',
             'n_cells', 'area_m2', 'n_parts', 'keep', 'ds_rank']
    rest = [c for c in gdf.columns if c not in order + ['geometry']]
    gdf = gdf[order + rest + ['geometry']]
    assert all(len(c) <= 10 for c in gdf.columns), 'DBF field name > 10'
    gdf.to_file(args.out_shp)

    # Sidecar: the authoritative node -> section table.
    map_csv = os.path.splitext(args.out_shp)[0] + '_node_map.csv'
    node_tab.sort_values(['reach_id', 'order'])[
        ['node_id', 'reach_id', 'flex_id', 'flex_num', 'observed',
         'p_length', 'p_dist_out']].to_csv(map_csv, index=False)

    # 6. Report.
    km = gdf['flex_len'] / 1000.0
    print('wrote {} flexible reaches (target {:g} km) from {} SWORD reaches '
          'to {}'.format(len(gdf), args.length_km, gdf.reach_id.nunique(),
                         args.out_shp))
    print('wrote node -> flex_id table to {}'.format(map_csv))
    print('  prior length: min {:.2f}  median {:.2f}  max {:.2f} km'.format(
        km.min(), km.median(), km.max()))
    print('  nodes per section: min {}  median {:.0f}  max {}'.format(
        gdf.n_nodes_p.min(), gdf.n_nodes_p.median(), gdf.n_nodes_p.max()))
    whole = gdf[gdf.n_flex == 1]
    if len(whole):
        print('  {} reach(es) were kept whole (no split brings them closer '
              'to {:g} km)'.format(whole.reach_id.nunique(), args.length_km))
    if target_m < 400:
        print('  note: {:g} km is about one or two SWORD nodes -- '
              '1.3_true_node_polygons.py may be what you want.'.format(
                  args.length_km))
    partial = gdf[gdf.obs_frac_n < 0.5]
    if len(partial):
        print('  {} section(s) observed on under half their nodes (often at '
              'the AOI edge):'.format(len(partial)))
        for _, r in partial.sort_values('obs_frac_n').head(10).iterrows():
            print('     {}  obs_frac_n {:.2f} ({} of {} nodes)'.format(
                r.flex_id, r.obs_frac_n, r.n_nodes_ob, r.n_nodes_p))
    multi = gdf[gdf.n_parts > 1]
    if len(multi):
        print('  {} section(s) are multipart (bars or gaps in the '
              'mask)'.format(len(multi)))
    _nodes.report_provenance(args.pixcvec_nc)
    _nodes.report_coverage(args.water_tif, n_assigned, n_excluded)
    if 'river_name' in gdf.columns:
        _reaches.report_river_mix(gdf)
    print('\nNEXT: open in QGIS over 1.4\'s reach polygons; section edges '
          'should fall on node\nboundaries and never cross a reach boundary. '
          'Order by ds_rank for a profile.')


if __name__ == '__main__':
    main()

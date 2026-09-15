#!/usr/bin/env python
"""Run each water mask through RiverObs.

Every (water mask, SWORD version) pair is passed to the patched
calval2rivertile.py, and the resulting RiverTile netCDF is returned
as two CSVs (one node-level, one reach-level).

Run from inside a RiverObs conda env, in the same folder
as run_calval2rivertile.py, riverobs_shim.py and reach_exclusions.py.

Usage:
    python 1.2_run_RiverObs.py \\
        --riverobs-root /path/to/RiverObs \\
        --raster-dir /path/rasterized_water_masks \\
        --rdf-dir . \\
        --out-dir /path/RiverObs_output \\
        --name YR_AOI \\
        --res 10

`--name` is the basename given to 1.1_rasterize_watermask.py, and may be
repeated to process several masks in one call:

    --name upperYR_071024 lowerYR_071624 \\
    --date 2024-07-10      2024-07-16

`--date` is optional and carries the matching SWOT overpass date (UTC)
through to the SWOT_date column of the output CSVs.

To leave SWORD reaches out of the run (e.g. a parallel channel at a
bifurcation) name them with `--exclude-reaches`:

    --exclude-reaches 81270500291
    --exclude-reaches exclude_YR_AOI.txt

Excluded reaches are removed from the prior database before RiverObs
runs, so they appear in neither the RiverTile, the PIXCVec.

Requires: netCDF4, pandas, numpy (all in the RiverObs environment)

-----------------------------------------------------------------------------
Script by:
Camryn Kluetmeier (camryn.kluetmeier@duke.edu)

Parts of this script were developed with assistance from Claude Code
(Anthropic) for debugging, documentation, and related editorial suggestions.

Last updated: 2026-09-15

"""

PIPELINE_VERSION = '2.0.0'


import argparse
import datetime
import os
import subprocess
import sys

import numpy as np
import netCDF4
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reach_exclusions


SWORD_VERSIONS = {
    'v16':  'riverobs_ortho_v16.rdf',   # pair with RiverSP version C / PIC0
    'v17b': 'riverobs_ortho_v17b.rdf',  # pair with RiverSP version D / PGD0
}

# Node fields worth keeping. `width` is the ortho width: RiverObs
# computed it as node area / SWORD p_length using the same
# RiverNode.width_area() call the operational processor uses for
# RiverSP.
NODE_FIELDS = [
    'reach_id', 'node_id', 'lat', 'lon',
    'width', 'width_u', 'area_total', 'area_detct',
    'n_good_pix', 'p_length', 'p_width', 'p_dist_out',
]

# Reach fields. `width` here is sum(node area) / sum(node p_length) over
# the observed nodes -- a length-weighted mean, not the unweighted mean
# of node widths. `obs_frac_n` is the fraction of the reach's SWORD
# nodes that received data, and is the defensible basis for a
# reach-completeness threshold. `p_dist_out` is the SWORD distance to
# outlet and is what orders reaches along stream.
REACH_FIELDS = [
    'reach_id', 'width', 'width_u', 'area_total', 'area_detct',
    'obs_frac_n', 'partial_f', 'n_good_nod', 'p_length', 'p_width',
    'p_dist_out',
]


def flatten_group(nc_path, group, fields):
    """Read one group of a RiverTile netCDF into a DataFrame."""
    out = {}
    with netCDF4.Dataset(nc_path, 'r') as ds:
        grp = ds.groups[group]
        for f in fields:
            if f not in grp.variables:
                print('    note: {} not in /{} -- skipping'.format(f, group))
                continue
            arr = grp.variables[f][:]
            if isinstance(arr, np.ma.MaskedArray):
                arr = arr.filled(np.nan) if arr.dtype.kind == 'f' else arr.filled(-999)
            out[f] = np.asarray(arr).ravel()
    return pd.DataFrame(out)


def run_one(survey, version, rdf, args):
    name = survey['name']
    res_tag = ('{:g}'.format(args.res)).replace('.', 'p')
    water_tif = os.path.join(
        args.raster_dir, '{}_water_{}m.tif'.format(name, res_tag))
    if not os.path.exists(water_tif):
        # FileNotFoundError, not SystemExit: SystemExit does not inherit
        # from Exception, so --keep-going cannot catch it and a single
        # missing raster would end the batch before any CSV is written.
        raise FileNotFoundError(
            'missing water raster: {}\n  --res here must match the --res '
            'given to 1.1_rasterize_watermask.py for {}'.format(
                water_tif, name))

    out_dir = os.path.join(args.out_dir, version)
    os.makedirs(out_dir, exist_ok=True)
    rivertile_nc = os.path.join(out_dir, 'ortho_{}_{}.nc'.format(name, version))
    pixcvec_nc = os.path.join(out_dir, 'ortho_{}_{}_pixcvec.nc'.format(name, version))
    pixc_nc = os.path.join(out_dir, 'ortho_{}_{}_pixc.nc'.format(name, version))

    # run_calval2rivertile.py, not src/bin/calval2rivertile.py -- the
    # wrapper loads riverobs_shim first. See riverobs_shim.py for what
    # it patches.
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'run_calval2rivertile.py'),
        water_tif, 'airborne_watermask',
        rivertile_nc, pixcvec_nc, rdf, pixc_nc,
        '--riverobs-root', args.riverobs_root,
        '--log-level', args.log_level,
    ]
    # Corridor factors and the exclusion list, if given, are forwarded
    # rather than relied on through the environment, so the exact command
    # printed below is reproducible.
    if args.wth_coef_factor is not None:
        cmd += ['--wth-coef-factor', repr(args.wth_coef_factor)]
    if args.ext_dist_coef_factor is not None:
        cmd += ['--ext-dist-coef-factor', repr(args.ext_dist_coef_factor)]
    if args.excluded:
        # Already resolved to explicit ids in main(), so the printed
        # command reproduces the run even if the .txt file changes later.
        cmd += ['--exclude-reaches', reach_exclusions.to_env(args.excluded)]
    print('\n=== {} / SWORD {} ==='.format(name, version))
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)

    nodes = flatten_group(rivertile_nc, 'nodes', NODE_FIELDS)
    reaches = flatten_group(rivertile_nc, 'reaches', REACH_FIELDS)

    for df in (nodes, reaches):
        df.insert(0, 'survey', name)
        df.insert(1, 'SWOT_date', survey['date'])
        df.insert(2, 'sword_version', version)

    # RiverObs writes 'width'; rename so downstream code cannot confuse
    # the reference width with the SWOT width column.
    nodes = nodes.rename(columns={
        'width': 'ortho_width_m', 'width_u': 'ortho_width_u_m',
        'area_total': 'ortho_area_total_m2', 'area_detct': 'ortho_area_detct_m2'})
    reaches = reaches.rename(columns={
        'width': 'ortho_width_m', 'width_u': 'ortho_width_u_m',
        'area_total': 'ortho_area_total_m2', 'area_detct': 'ortho_area_detct_m2'})

    print('    {} nodes, {} reaches'.format(len(nodes), len(reaches)))
    if args.excluded:
        leaked = sorted(set(reaches['reach_id']).intersection(args.excluded))
        if leaked:
            # Should be impossible -- the shim drops these before
            # assignment -- so say so loudly rather than let a
            # "single-channel" CSV quietly contain both channels.
            print('    WARNING: excluded reach(es) {} are STILL in the '
                  'RiverTile. The exclusion did not take effect; do not '
                  'treat this output as single-channel.'.format(
                      ', '.join(str(r) for r in leaked)))
    return nodes, reaches


def build_surveys(names, dates):
    """
    Pair --name with --date.

    One date for all names, one date per name, or no dates at all. Anything
    else is a length mismatch that would silently attach the wrong overpass
    date to a survey, so it is an error rather than a zip() truncation.
    """
    if not dates:
        return [{'name': n, 'date': ''} for n in names]
    if len(dates) == 1 and len(names) > 1:
        dates = dates * len(names)
    if len(dates) != len(names):
        raise SystemExit(
            '--date takes one value per --name ({} given for {} name(s)), or '
            'a single value for all of them, or nothing at all.\n  names: '
            '{}\n  dates: {}'.format(
                len(dates), len(names), ', '.join(names), ', '.join(dates)))

    bad = []
    for name, date in zip(names, dates):
        try:
            datetime.datetime.strptime(date, '%Y-%m-%d')
        except (ValueError, TypeError):
            bad.append('{}: {!r}'.format(name, date))
    if bad:
        raise SystemExit(
            '--date must be ISO YYYY-MM-DD (the SWOT overpass date, not the '
            'flight date).\n  ' + '\n  '.join(bad))
    return [{'name': n, 'date': d} for n, d in zip(names, dates)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--riverobs-root', required=True,
                   help='root of your RiverObs fork (the dir containing src/)')
    p.add_argument('--raster-dir', required=True,
                   help='output directory from 1.1_rasterize_watermask.py')
    p.add_argument('--rdf-dir', required=True,
                   help='directory holding riverobs_ortho_v16.rdf / _v17b.rdf')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--name', nargs='+', required=True, metavar='NAME',
                   help='the --name given to 1.1_rasterize_watermask.py. '
                        'Repeatable to process several masks in one call.')
    p.add_argument('--date', nargs='+', default=None, metavar='YYYY-MM-DD',
                   help='SWOT overpass date (UTC) for each --name, or one '
                        'date for all of them. Copied to the SWOT_date '
                        'column; left empty if omitted.')
    p.add_argument('--res', type=float, default=3.0,
                   help='must match the --res used in '
                        '1.1_rasterize_watermask.py')
    p.add_argument('--versions', nargs='+', default=['v16', 'v17b'])
    p.add_argument('--keep-going', action='store_true',
                   help='carry on if a survey fails, and still write the CSVs '
                        'for the ones that worked. Failures are listed at the '
                        'end and the exit status is non-zero.')
    p.add_argument('--log-level', default='info')
    p.add_argument('--wth-coef-factor', type=float, default=None,
                   help='multiply the prior wth_coef, widening BOTH search-'
                        'corridor gates. 1.0 = stock RiverObs. Overrides '
                        'RIVEROBS_WTH_COEF_FACTOR.')
    p.add_argument('--ext-dist-coef-factor', type=float, default=None,
                   help='multiply the prior ext_dist_coef (gate 2). 1.0 = stock '
                        'RiverObs. Overrides RIVEROBS_EXT_DIST_COEF_FACTOR.')
    reach_exclusions.add_argument(p)
    p.add_argument('--out-tag', default=None,
                   help='suffix for the combined CSV filenames, e.g. '
                        '--out-tag single_channel writes '
                        'ortho_riverobs_nodes_single_channel.csv. Use it so an '
                        'excluded run does not overwrite the full one.')
    args = p.parse_args()
    print('# {} {}'.format(os.path.basename(__file__), PIPELINE_VERSION))

    unknown = [v for v in args.versions if v not in SWORD_VERSIONS]
    if unknown:
        raise SystemExit('unknown --versions {}. Known: {}'.format(
            ', '.join(unknown), ', '.join(sorted(SWORD_VERSIONS))))

    try:
        args.excluded = reach_exclusions.parse(args.exclude_reaches)
    except reach_exclusions.ExclusionError as exc:
        raise SystemExit('--exclude-reaches: {}'.format(exc))
    if args.excluded:
        print('excluding {} from the prior database'.format(
            reach_exclusions.describe(args.excluded, limit=1000)))

    surveys = build_surveys(args.name, args.date)
    if not any(s['date'] for s in surveys):
        print('note: no --date given, so the SWOT_date column will be empty. '
              'Add --date YYYY-MM-DD if anything downstream joins on it.')

    all_nodes, all_reaches, failures = [], [], []
    for version in args.versions:
        rdf = os.path.join(args.rdf_dir, SWORD_VERSIONS[version])
        for survey in surveys:
            try:
                nodes, reaches = run_one(survey, version, rdf, args)
            except (Exception, SystemExit) as exc:
                if not args.keep_going:
                    raise
                failures.append((survey['name'], version, str(exc).split(chr(10))[0]))
                print('    FAILED: {} / {} -- {}'.format(
                    survey['name'], version, type(exc).__name__))
                continue
            all_nodes.append(nodes)
            all_reaches.append(reaches)

    if not all_nodes:
        raise SystemExit('\nevery run failed -- nothing to write')

    os.makedirs(args.out_dir, exist_ok=True)
    tag = '_' + args.out_tag if args.out_tag else '_all'
    node_csv = os.path.join(
        args.out_dir, 'ortho_riverobs_nodes{}.csv'.format(tag))
    reach_csv = os.path.join(
        args.out_dir, 'ortho_riverobs_reaches{}.csv'.format(tag))
    pd.concat(all_nodes, ignore_index=True).to_csv(node_csv, index=False)
    pd.concat(all_reaches, ignore_index=True).to_csv(reach_csv, index=False)
    print('\nwrote', node_csv)
    print('wrote', reach_csv)

    if failures:
        print('\n{} run(s) FAILED and are absent from those CSVs:'.format(
            len(failures)))
        for name, version, msg in failures:
            print('   {} / {}'.format(name, version))
        print('Re-run just those with --name <name> once fixed.')
        sys.exit(1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""Run calval2rivertile.py with the runtime shim loaded.

Usage:
    python run_calval2rivertile.py \\
        upperYR_071024_water_3m.tif airborne_watermask \\
        out_rivertile.nc out_pixcvec.nc riverobs_ortho_v17b.rdf \\
        out_pixc.nc \\
        --riverobs-root "/path/to/RiverObs" --log-level info

`--riverobs-root` is optional if PYTHONPATH already points at
RiverObs/src. Keep this file in the same folder as riverobs_shim.py and
reach_exclusions.py.

The search-corridor factors are set here, because this is where the
pixel-to-node assignment happens:

    --wth-coef-factor 3.0 --ext-dist-coef-factor 3.0

So is the reach exclusion list, for the same reason -- excluding a reach
removes it from the prior database before assignment:

    --exclude-reaches 81270500291
    --exclude-reaches exclude_YR_AOI.txt

Both are recorded on the RiverTile and PIXCVec as global attributes, so a
finished product says which corridor and which exclusion list produced it.

Excluding a reach here CHANGES PIXEL ASSIGNMENT for its neighbours as
well: RiverObs gives every water cell to its single nearest reach, so the
excluded channel's water is freed for whatever kept reach is within
corridor range. riverobs_shim.py patch 6 has the full argument, and
reports the nearest kept reach and its half-corridor for each exclusion.
To drop a parallel channel from the OUTPUT without disturbing the stem's
footprint, leave it in here and use --exclude-reaches on
1.3_true_node_polygons.py / 1.4_true_reach_polygons.py instead.

-----------------------------------------------------------------------------
Script by:
Camryn Kluetmeier (camryn.kluetmeier@duke.edu)

Parts of this script were developed with assistance from Claude Code
(Anthropic) for debugging, documentation, and related editorial suggestions.

Last updated: 2026-09-15

"""

PIPELINE_VERSION = '1.2.0'


import os
import sys

# reach_exclusions.py and riverobs_shim.py live next to this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reach_exclusions

# --riverobs-root is ours, not calval2rivertile's -- pull it out of argv
# first.
_root = None
if '--riverobs-root' in sys.argv:
    i = sys.argv.index('--riverobs-root')
    _root = os.path.expanduser(sys.argv[i + 1])
    del sys.argv[i:i + 2]

# Likewise the corridor factors. These must be settled BEFORE
# riverobs_shim is imported, because the shim reads the environment at
# import time.
_factors = {}
for _flag, _env in (('--wth-coef-factor', 'RIVEROBS_WTH_COEF_FACTOR'),
                    ('--ext-dist-coef-factor', 'RIVEROBS_EXT_DIST_COEF_FACTOR')):
    if _flag in sys.argv:
        i = sys.argv.index(_flag)
        try:
            _val = float(sys.argv[i + 1])
        except (IndexError, ValueError):
            sys.exit('{} needs a number, e.g. {} 3.0'.format(_flag, _flag))
        if _val <= 0:
            sys.exit('{} must be positive (1.0 = stock RiverObs)'.format(_flag))
        del sys.argv[i:i + 2]
        os.environ[_env] = repr(_val)
        _factors[_env] = _val

# And the exclusion list, for the same reason. Repeatable, and each
# occurrence takes one or more values (ids and/or .txt paths) up to the
# next flag -- the same shape argparse would give it with nargs='+',
# hand-rolled because calval2rivertile.py owns the real parser and would
# reject an argument it does not know.
_exclude_values = []
while '--exclude-reaches' in sys.argv:
    i = sys.argv.index('--exclude-reaches')
    j = i + 1
    while j < len(sys.argv) and not sys.argv[j].startswith('--'):
        _exclude_values.append(sys.argv[j])
        j += 1
    if j == i + 1:
        sys.exit('--exclude-reaches needs at least one reach id or a .txt '
                 'file, e.g. --exclude-reaches 81270500291')
    del sys.argv[i:j]

try:
    _excluded = reach_exclusions.parse(_exclude_values)
except reach_exclusions.ExclusionError as exc:
    sys.exit('--exclude-reaches: {}'.format(exc))
if _excluded:
    os.environ['RIVEROBS_EXCLUDE_REACHES'] = reach_exclusions.to_env(_excluded)

if _root:
    src = os.path.join(_root, 'src')
    if not os.path.isdir(src):
        sys.exit('no src/ under {}'.format(_root))
    sys.path.insert(0, src)

try:
    import SWOTRiver  # noqa: F401
except ImportError:
    sys.exit(
        'Cannot import SWOTRiver. Either set PYTHONPATH to RiverObs/src or '
        'pass --riverobs-root /path/to/RiverObs')

import riverobs_shim  # noqa: F401  (installs the patches on import)

# calval2rivertile's main() is located and executed in this process,
# which riverobs_shim has already patched.
import importlib.util

_cv_path = None
for entry in sys.path:
    candidate = os.path.join(entry, '..', 'src', 'bin', 'calval2rivertile.py')
    candidate = os.path.normpath(candidate)
    if os.path.exists(candidate):
        _cv_path = candidate
        break
if _cv_path is None and _root:
    _cv_path = os.path.join(_root, 'src', 'bin', 'calval2rivertile.py')
if _cv_path is None or not os.path.exists(_cv_path):
    import SWOTRiver as _sr
    guess = os.path.normpath(os.path.join(
        os.path.dirname(_sr.__file__), '..', 'bin', 'calval2rivertile.py'))
    _cv_path = guess

if not os.path.exists(_cv_path):
    sys.exit('could not find calval2rivertile.py; pass --riverobs-root')

_spec = importlib.util.spec_from_file_location('calval2rivertile', _cv_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

def _stamp_provenance(paths):
    """Record the corridor factors and exclusion list on the outputs as
    global attributes.

    Purely additive, and never allowed to fail a run that has already produced
    its products -- a netCDF that cannot be reopened for append is worth a
    warning, not a lost result.

    riverobs_excluded_reaches is what the shim ACTUALLY dropped, which is
    not always what was asked for -- an id outside the bounding box has no
    effect. Both are stamped, under separate names, so a product can be
    read back without having to trust that the two agreed.
    """
    import netCDF4
    applied, not_found = riverobs_shim.excluded_reaches_applied()
    attrs = {
        'riverobs_wth_coef_factor': riverobs_shim.WTH_COEF_FACTOR,
        'riverobs_ext_dist_coef_factor': riverobs_shim.EXT_DIST_COEF_FACTOR,
        'riverobs_shim_version': riverobs_shim.PIPELINE_VERSION,
        'riverobs_excluded_reaches_requested':
            reach_exclusions.to_env(_excluded),
        'riverobs_excluded_reaches': reach_exclusions.to_env(applied),
        'riverobs_excluded_reaches_not_in_tile':
            reach_exclusions.to_env(not_found),
        'riverobs_exclusion_mode': 'prior_database' if _excluded else 'none',
    }
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        try:
            with netCDF4.Dataset(path, 'a') as ds:
                for k, v in attrs.items():
                    setattr(ds, k, v)
        except Exception as exc:
            print('note: could not stamp provenance on {}: {}: {}'
                  .format(os.path.basename(path), type(exc).__name__, exc))


if __name__ == '__main__':
    # Positional order is fixed by calval2rivertile.py's own parser:
    #   input_file format out_riverobs_file out_pixcvec_file rdf_file [out_pixc_file]
    # Strip the one optional flag it takes so the positions hold no
    # matter where the caller put it. Our own flags are already gone from
    # argv by this point.
    _pos, _skip = [], False
    for _a in sys.argv[1:]:
        if _skip:
            _skip = False
            continue
        if _a in ('-l', '--log-level'):
            _skip = True
            continue
        if _a.startswith('-') and len(_a) > 1:
            continue
        _pos.append(_a)
    _outputs = _pos[2:4]

    if _factors:
        print('corridor factors: wth_coef x{}, ext_dist_coef x{}'.format(
            riverobs_shim.WTH_COEF_FACTOR, riverobs_shim.EXT_DIST_COEF_FACTOR))
    if _excluded:
        print('excluding from the prior database: {}'.format(
            reach_exclusions.describe(_excluded, limit=1000)))
        print('  (this frees their water for neighbouring reaches -- see the')
        print('   "nearest kept reach" lines in the log below)')

    _mod.main()
    _stamp_provenance(_outputs)

    _applied, _not_found = riverobs_shim.excluded_reaches_applied()
    if _excluded:
        print('\nexcluded from the prior database: {}'.format(
            reach_exclusions.describe(_applied, limit=1000)))
        if _not_found:
            print('NOT in this bounding box, so ignored: {}'.format(
                reach_exclusions.to_env(_not_found)))
        if not _applied and not _not_found:
            # Patch 6 reports through this pair every time ReachExtractor
            # runs, so both being empty means it never ran -- the outputs
            # are a full-network run wearing an exclusion flag. Say so
            # rather than let that pass as success.
            print('WARNING: the exclusion list never reached the prior '
                  'database. Patch 6 did not run,')
            print('         so these outputs include every SWORD reach in '
                  'the bounding box. Check that')
            print('         riverobs_shim is the one next to this script and '
                  'that its version is >= 1.2.0.')

    absent = riverobs_shim.absent_variables()
    if absent:
        print('\nshim substituted missing_value for these pixel variables: {}'
              .format(', '.join(absent)))
        print('(time_from_prev_xover / time_to_next_xover are expected; '
              'anything else is worth a look)')

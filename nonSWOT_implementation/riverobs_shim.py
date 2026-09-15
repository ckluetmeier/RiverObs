"""Runtime fixes for the RiverObs CalVal path.

Six fixes:

  1. SWOTRiverEstimator.__init__
     CalValToRiverTile never passes reach_pct_good_sus_thresh
     (L2PixcToRiverTile does), so it arrives as None and
     get_reach_mask() raises
       TypeError: '<=' not supported between instances of 'float'
       and 'NoneType'
     The shim substitutes 0. Every water-mask cell is good, so the
     percentage being tested is always 100 and any value below that
     is neutral.

  2. RiverReach.__init__
     Under height_agg_method='orig', wse_s_u is a scalar constant
     rather than a per-node array, and the output packing loop
     raises
       ValueError: zero-dimensional arrays cannot be concatenated
     A single-node reach hits the same wall from the other side,
     with a different message:
       ValueError: all the input arrays must have same number of
       dimensions, but the array at index 0 has 1 dimension(s) and
       the array at index N has 0 dimension(s)
     Estimate.py treats every RiverReach attribute except
     ds/metadata as a NODE variable and feeds it to np.concatenate,
     which rejects anything 0-d no matter how many nodes the reach
     has. The shim broadcasts any scalar attribute -- 0-d ndarray,
     np.float64 and friends, or a plain Python number -- to the
     reach's node count, including when that count is 1. Real
     per-node arrays and string attributes are untouched.

  3. ReachExtractor.__init__
     SWORD reaches have different numbers of centerline points. The
     output packing loop does
     np.array([reach.metadata[k] for reach in ...]) over
     centerline_lon / centerline_lat, and since numpy 1.24 a ragged
     list raises
       ValueError: setting an array element with a sequence. The
       requested array has an inhomogeneous shape after 1
       dimensions.
     The RiverTile product declares centerline_lat/lon as 2-D
     (reaches, centerlines), so a rectangular array is what it needs
     anyway. The shim pads every reach's centerline arrays to the
     longest one using the product's own fill value. rivertile.py
     already filters the padding out when it builds the reach
     LineString -- `is_valid = np.abs(lats) < 90`.

  4. ReachExtractor.__init__ -- corridor relaxation (opt-in,
     default off)
     RiverObs sizes its cross-channel search corridor from the prior
     database's channel width. Two gates apply, both scaled by the
     per-node wth_coef and ext_dist_coef:

       gate 1  SWOTRiverEstimator.assign_reaches sets
                   search_width = 2 * prior_width * wth_coef
               and RiverObs.get_ext_dist_threshold takes
                   half-corridor = search_width / 3
                                 = 0.667 * prior_width * wth_coef

       gate 2  SWOTRiverEstimator.assign_reaches_ext_dist_coef
               re-masks with
                   extreme_dist = ext_dist_coef
                                  * max(node_spacing,
                                        max(prior_max_width,
                                            prior_width)
                                        * wth_coef)

     Setting RIVEROBS_WTH_COEF_FACTOR /
     RIVEROBS_EXT_DIST_COEF_FACTOR (or calling
     set_corridor_factors()) multiplies those two coefficients,
     which widens both gates together.

  5. RiverObs.get_node_stat
     time_from_prev_xover / time_to_next_xover are per-line PIXC
     variables a water mask does not have. SWOTRiverEstimator skips
     loading them, then asks for their node means anyway, raising
       AttributeError: 'RiverNode' object has no attribute
       'time_from_prev_xover'
     The shim returns the missing-value sentinel for exactly these
     two and re-raises for anything else, so RiverObs's own
     AttributeError handlers (sig0, geoid, tides, layover_impact,
     bright_land_flag) still run as upstream wrote them.

  6. ReachExtractor.__init__ -- reach exclusion (opt-in,
     default off)
     SWORD represents some bifurcations as two parallel reaches over
     two channels. An along-stream profile tied to SWORD wants one
     of them, not both: the parallel reach puts a second value at
     the same p_dist_out and its nodes double back across the
     floodplain.

     Setting RIVEROBS_EXCLUDE_REACHES (or calling
     set_reach_exclusions()) removes the named reach ids from the
     prior database right after ReachExtractor reads it, so nothing
     downstream ever sees them -- not the RiverTile, not the
     PIXCVec, not the CSVs. See reach_exclusions.py for the accepted
     forms of the list.

     THIS CHANGES PIXEL ASSIGNMENT, and not only for the excluded
     reach. SWOTRiverEstimator.assign_reaches (~line 1065) makes the
     assignment competitive and unique: it builds one Centerline
     from every reach's nodes, takes each pixel's distance to that
     tile-wide centerline as the target

         min_dist = tile_d

     and then lets a reach keep a pixel only when it does better,

         mask = river_obs.d <= these_min_dists

     so each water cell ends up with the single nearest reach. Take
     a reach out of the prior and its channel's water is not
     discarded -- it is simply freed for whatever kept reach comes
     next inside its corridor. The neighbour's width, area and node
     footprints then grow across the bifurcation.

     Whether that happens depends on the corridor, so the shim
     measures it: for each excluded reach it reports the nearest
     kept reaches, their distance in metres, and the half-corridor
     from patch 4's two gates. A kept reach closer than its own
     half-corridor will absorb the excluded channel. At
     wth_coef x5 / ext_dist_coef x5 on a braided river, that is the
     expected case, not the exception.

     If you want the parallel channel gone from the OUTPUT while
     leaving the stem's footprint exactly as this run computed it,
     do not exclude it here. Let the run keep both reaches -- so the
     parallel reach goes on claiming its own water -- and drop it
     afterwards with --exclude-reaches on 1.3_true_node_polygons.py
     / 1.4_true_reach_polygons.py, which filters the finished
     PIXCVec without touching the assignment.

     Excluded ids that are not in the tile are reported and ignored,
     not treated as an error: one exclusion file is meant to be
     usable across AOIs, where most of its ids are legitimately out
     of frame. Check the report if you expected a reach to go.

-----------------------------------------------------------------------------
Script by:
Camryn Kluetmeier (camryn.kluetmeier@duke.edu)

Parts of this script were developed with assistance from Claude Code
(Anthropic) for debugging, documentation, and related editorial suggestions.

Last updated: 2026-09-15

"""

PIPELINE_VERSION = '1.2.0'


import logging
import os
import sys

import numpy as np

# reach_exclusions.py lives next to this file. run_calval2rivertile.py puts
# that directory on sys.path before importing the shim, but the shim is also
# imported directly in notebooks, so make the import work either way.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reach_exclusions

LOGGER = logging.getLogger('riverobs_shim')

_ABSENT_REPORTED = set()

# Only these are substituted. Every other missing variable is re-raised
# so that RiverObs's own try/except handlers (sig0, geoid, tides,
# layover_impact, bright_land_flag) run exactly as upstream designed
# them to. Intercepting those would replace upstream's NaN fill with the
# missing-value sentinel and quietly change what lands in the output
# product.
SUBSTITUTABLE = ('time_from_prev_xover', 'time_to_next_xover')

# Corridor relaxation. 1.0 = stock RiverObs behaviour. Set with the
# environment variables below, or by calling set_corridor_factors()
# before processing.
#
#   RIVEROBS_WTH_COEF_FACTOR       multiplies wth_coef      (both gates)
#   RIVEROBS_EXT_DIST_COEF_FACTOR  multiplies ext_dist_coef (gate 2, and the
#                                  dominant-label extension in gate 1)
#
# Raising these admits more water per node. A corridor wider than needed
# lets adjacent reaches claim the same water (cross-reach double
# counting), which is the failure mode to watch for.
WTH_COEF_FACTOR = float(os.environ.get('RIVEROBS_WTH_COEF_FACTOR', 1.0))
EXT_DIST_COEF_FACTOR = float(os.environ.get('RIVEROBS_EXT_DIST_COEF_FACTOR', 1.0))

# Reach exclusion (patch 6). Empty = stock RiverObs behaviour: every reach
# in the bounding box is processed. Set with RIVEROBS_EXCLUDE_REACHES (a
# comma-separated id list or a path to a .txt file) or by calling
# set_reach_exclusions() before processing.
EXCLUDE_REACHES = reach_exclusions.from_env(
    'RIVEROBS_EXCLUDE_REACHES',
    warn=lambda msg: LOGGER.warning('shim: %s', msg))

# Filled in by patch 6 so the caller can report what actually happened
# rather than what was asked for.
_EXCLUDED_APPLIED = []
_EXCLUDED_NOT_FOUND = []


def set_corridor_factors(wth=None, ext_dist=None):
    """Override the corridor factors programmatically. Call before processing."""
    global WTH_COEF_FACTOR, EXT_DIST_COEF_FACTOR
    if wth is not None:
        WTH_COEF_FACTOR = float(wth)
    if ext_dist is not None:
        EXT_DIST_COEF_FACTOR = float(ext_dist)


def set_reach_exclusions(reaches=None):
    """
    Override the exclusion list programmatically. Call before processing.

    `reaches` takes any form reach_exclusions.parse accepts: a list of
    ids, a comma-separated string, or a path to a .txt file. None or an
    empty value clears the list.
    """
    global EXCLUDE_REACHES
    EXCLUDE_REACHES = reach_exclusions.parse(reaches)
    return EXCLUDE_REACHES


def excluded_reaches_applied():
    """(applied, not_found) -- what patch 6 actually dropped, this session.

    `applied` is the ids that were present in the tile and removed.
    `not_found` is the ids that were asked for but are not in this
    bounding box. Empty lists before ReachExtractor has run.
    """
    return list(_EXCLUDED_APPLIED), list(_EXCLUDED_NOT_FOUND)


# The RiverTile product's declared fill for centerline_lat/lon. Chosen
# so that rivertile.py's own `np.abs(lats) < 90` test drops the padding.
MISSING_VALUE_FLT = -999999999999.0


def install():
    """Apply all patches. Idempotent."""
    _patch_estimator_init()
    _patch_river_reach()
    _patch_reach_extractor()
    _patch_get_node_stat()
    LOGGER.info('riverobs_shim %s installed (6 runtime patches)',
                PIPELINE_VERSION)


# Patches 3, 4 and 6: ReachExtractor.__init__
def _patch_reach_extractor():
    from RiverObs import ReachDatabase as _RD
    klass = _RD.ReachExtractor
    if getattr(klass.__init__, '_shimmed', False):
        return
    original = klass.__init__

    KEYS = ('centerline_lon', 'centerline_lat')

    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)

        # Order matters. The corridor factors go on first because patch 6's
        # report quotes the resulting half-corridor; the exclusion runs
        # next so that the centerline padding below is sized over the
        # reaches that survive.
        _apply_corridor_factors(self)
        _apply_reach_exclusions(self)

        reaches = getattr(self, 'reach', None) or []

        lengths = [len(np.atleast_1d(r.metadata[k]))
                   for r in reaches for k in KEYS if k in r.metadata]
        if not lengths or len(set(lengths)) == 1:
            return                      # already rectangular, nothing to do
        n_max = max(lengths)
        for r in reaches:
            for k in KEYS:
                if k not in r.metadata:
                    continue
                v = np.atleast_1d(r.metadata[k])
                v = np.ma.filled(np.ma.masked_invalid(
                    np.asarray(v, dtype='f8')), MISSING_VALUE_FLT)
                if len(v) < n_max:
                    v = np.concatenate(
                        [v, np.full(n_max - len(v), MISSING_VALUE_FLT)])
                r.metadata[k] = v
        LOGGER.info('shim: padded centerline_lon/lat across %d reaches to %d '
                    'points (was %d-%d) so the reach output packs to a '
                    'rectangular array', len(reaches), n_max,
                    min(lengths), max(lengths))

    __init__._shimmed = True
    klass.__init__ = __init__


# Patch 1: SWOTRiverEstimator.__init__
def _patch_estimator_init():
    import SWOTRiver
    klass = SWOTRiver.SWOTRiverEstimator
    if getattr(klass.__init__, '_shimmed', False):
        return
    original = klass.__init__

    def __init__(self, *args, **kwargs):
        if kwargs.get('reach_pct_good_sus_thresh') is None:
            kwargs['reach_pct_good_sus_thresh'] = 0
            LOGGER.debug('shim: reach_pct_good_sus_thresh None -> 0')
        return original(self, *args, **kwargs)

    __init__._shimmed = True
    klass.__init__ = __init__


# Patch 2: RiverReach.__init__
def _patch_river_reach():
    from RiverObs.RiverReach import RiverReach
    if getattr(RiverReach.__init__, '_shimmed', False):
        return
    original = RiverReach.__init__

    def __init__(self, **kwds):
        original(self, **kwds)
        lat = getattr(self, 'lat', None)
        if lat is None:
            return
        n_nodes = len(np.atleast_1d(lat))
        if n_nodes < 1:
            return
        for key, value in list(self.__dict__.items()):
            if key in ('ds', 'metadata'):
                continue
            # Every attribute except ds/metadata is treated by
            # Estimate.py as a NODE variable and fed to np.concatenate,
            # which rejects anything 0-dimensional -- including a
            # single-node reach. Three shapes have to be caught:
            #   * a 0-d ndarray
            #   * np.float64 and friends, which are np.generic, NOT np.ndarray
            #   * a plain Python float/int
            # Strings and bytes are excluded: broadcasting one would be
            # wrong, and a string attribute would fail further
            # downstream anyway.
            is_zero_d = (
                (isinstance(value, np.ndarray) and value.ndim == 0) or
                (isinstance(value, np.generic)
                 and not isinstance(value, (np.str_, np.bytes_))) or
                isinstance(value, (float, int, bool)))
            if is_zero_d:
                setattr(self, key, np.repeat(np.asarray(value)[np.newaxis],
                                             n_nodes))
                LOGGER.debug('shim: broadcast scalar RiverReach.%s to %d nodes',
                             key, n_nodes)

    __init__._shimmed = True
    RiverReach.__init__ = __init__


# Patch 5: RiverObs.get_node_stat
def _patch_get_node_stat():
    from RiverObs.RiverObs import RiverObs
    if getattr(RiverObs.get_node_stat, '_shimmed', False):
        return
    original = RiverObs.get_node_stat

    def get_node_stat(self, stat, var, all_nodes=False, **kwargs):
        try:
            return original(self, stat, var, all_nodes=all_nodes, **kwargs)
        except AttributeError:
            if var not in SUBSTITUTABLE:
                raise
            if var not in _ABSENT_REPORTED:
                _ABSENT_REPORTED.add(var)
                LOGGER.warning(
                    'shim: pixel variable %r absent from this pixel cloud; '
                    'returning missing_value for its node statistics', var)
            n = (len(self.all_nodes) if all_nodes
                 else len(self.populated_nodes))
            return [self.missing_value] * n

    get_node_stat._shimmed = True
    RiverObs.get_node_stat = get_node_stat


def _half_corridor(reach, scalar_max_width=600.0):
    """
    The binding cross-channel half-corridor for one reach, in metres, after
    the patch-4 factors. Same two gates as _apply_corridor_factors; see the
    module docstring for where each comes from in RiverObs.

    Returns nan if the reach carries no usable prior width.
    """
    try:
        width = np.ma.filled(np.asarray(reach.width, dtype='f8'), np.nan)
        maxw = np.ma.filled(np.asarray(reach.max_width, dtype='f8'), np.nan)
        wth = np.asarray(reach.wth_coef, dtype='f8')
        ext = np.asarray(reach.ext_dist_coef, dtype='f8')
        node_len = np.asarray(reach.node_length, dtype='f8')
        with np.errstate(invalid='ignore'):
            gate1 = np.nanmedian(2.0 * width * wth / 3.0)
            gate2 = np.nanmedian(
                ext * np.maximum(node_len, np.fmax(maxw, width) * wth))
        return float(np.nanmin([gate1, gate2]))
    except Exception:
        return float('nan')


def _min_node_distance(reach_a, reach_b):
    """Smallest distance in metres between any node of A and any node of B.

    ReachExtractor has already projected node coordinates to the tile's
    metric CRS as .x / .y, so this is a plain Euclidean minimum.
    """
    try:
        ax = np.asarray(reach_a.x, dtype='f8').ravel()
        ay = np.asarray(reach_a.y, dtype='f8').ravel()
        bx = np.asarray(reach_b.x, dtype='f8').ravel()
        by = np.asarray(reach_b.y, dtype='f8').ravel()
        if not len(ax) or not len(bx):
            return float('nan')
        d2 = ((ax[:, None] - bx[None, :]) ** 2 +
              (ay[:, None] - by[None, :]) ** 2)
        return float(np.sqrt(d2.min()))
    except Exception:
        return float('nan')


# Patch 6: reach exclusion
def _apply_reach_exclusions(extractor, n_report=3):
    """
    Remove the excluded reach ids from a populated ReachExtractor, in place.

    ReachExtractor keeps two parallel sequences -- .reach (RiverReach
    objects) and .reach_idx (their SWORD reach ids) -- and everything
    downstream pairs them by position: assign_reaches does
    `for i_reach, reach_idx in enumerate(self.reaches.reach_idx)` and then
    indexes `self.reaches[i_reach]`. So they must be filtered in lockstep,
    and .nreaches (which __len__ and the iterator read) has to follow.

    .reach_index is renumbered to stay equal to the object's new position.
    Nothing in the CalVal path indexes by it -- WidthDataBase is the only
    consumer and use_width_db is False -- but leaving stale values behind
    would make the corridor log lines point at the wrong reach.
    """
    del _EXCLUDED_APPLIED[:]
    del _EXCLUDED_NOT_FOUND[:]

    if not EXCLUDE_REACHES:
        return

    ids = getattr(extractor, 'reach_idx', None)
    reaches = getattr(extractor, 'reach', None)
    if ids is None or reaches is None or len(ids) != len(reaches):
        LOGGER.error(
            'shim: cannot apply the reach exclusion list -- this '
            'ReachExtractor has %s reach ids for %s reaches. NOTHING WAS '
            'EXCLUDED; the run continues with every reach in the bounding '
            'box.',
            'no' if ids is None else len(ids),
            'no' if reaches is None else len(reaches))
        return

    wanted = set(int(r) for r in EXCLUDE_REACHES)
    present = {}
    for pos, rid in enumerate(ids):
        try:
            present[int(rid)] = pos
        except (TypeError, ValueError):
            continue

    drop_pos = {pos for rid, pos in present.items() if rid in wanted}
    dropped = sorted(rid for rid in present if rid in wanted)
    not_found = sorted(wanted - set(present))

    _EXCLUDED_NOT_FOUND.extend(not_found)
    if not_found:
        LOGGER.warning(
            'shim: %d excluded reach id(s) are not in this bounding box and '
            'had no effect: %s. If you expected one of these to be dropped, '
            'check it against the reach_id values in the RiverTile of an '
            'unfiltered run.',
            len(not_found), ', '.join(str(r) for r in not_found))

    if not dropped:
        LOGGER.warning(
            'shim: reach exclusion asked for %d reach(es), none of which are '
            'in this tile -- the run is identical to an unfiltered one',
            len(wanted))
        return

    if len(drop_pos) >= len(reaches):
        LOGGER.error(
            'shim: the exclusion list covers every one of the %d reaches in '
            'this bounding box. NOTHING WAS EXCLUDED -- an empty prior '
            'database would fail later with a much less obvious error. Check '
            'the list against the AOI.', len(reaches))
        return

    # Report before dropping, while both sides are still addressable.
    _report_exclusion_risk(extractor, drop_pos, n_report)

    keep = [pos for pos in range(len(reaches)) if pos not in drop_pos]
    extractor.reach = [reaches[pos] for pos in keep]
    extractor.reach_idx = [ids[pos] for pos in keep]
    extractor.nreaches = len(extractor.reach)
    extractor.idx = 0
    for new_pos, reach in enumerate(extractor.reach):
        try:
            reach.reach_index = new_pos
        except Exception:
            pass

    _EXCLUDED_APPLIED.extend(dropped)
    LOGGER.warning(
        'shim: EXCLUDED %d reach(es) from the prior database: %s. %d of %d '
        'reaches remain. Their water is now unclaimed and any kept reach '
        'whose corridor reaches it will take it -- compare reach width and '
        'area against an unfiltered run before trusting them.',
        len(dropped), ', '.join(str(r) for r in dropped),
        len(keep), len(reaches))


def _report_exclusion_risk(extractor, drop_pos, n_report):
    """
    For each reach about to be dropped, name the nearest kept reaches and
    say whether their corridor already reaches it.

    This is the diagnostic for the one real hazard in prior-level
    exclusion: assign_reaches gives every pixel to its nearest reach, so
    removing a reach hands its channel to whichever kept reach is close
    enough. "Close enough" is the patch-4 half-corridor, so distance and
    half-corridor are reported side by side.
    """
    reaches = extractor.reach
    ids = extractor.reach_idx
    keep_pos = [pos for pos in range(len(reaches)) if pos not in drop_pos]

    for pos in sorted(drop_pos):
        rid = ids[pos]
        ranked = sorted(
            ((_min_node_distance(reaches[pos], reaches[kpos]), kpos)
             for kpos in keep_pos),
            key=lambda pair: (np.isnan(pair[0]), pair[0]))

        up = _adjacent_ids(reaches[pos], 'rch_id_up')
        dn = _adjacent_ids(reaches[pos], 'rch_id_dn')
        LOGGER.warning(
            'shim: excluding reach %s (%d nodes, SWORD up %s / dn %s)',
            rid, len(np.atleast_1d(reaches[pos].x)),
            ', '.join(up) or '-', ', '.join(dn) or '-')

        for dist, kpos in ranked[:n_report]:
            half = _half_corridor(reaches[kpos])
            if np.isnan(dist) or np.isnan(half):
                verdict = 'cannot tell'
            elif dist <= half:
                verdict = 'WILL LIKELY ABSORB IT'
            else:
                verdict = 'probably out of reach'
            LOGGER.warning(
                'shim:   nearest kept reach %s -- %.0f m away, '
                'half-corridor %.0f m -> %s',
                ids[kpos], dist, half, verdict)


def _adjacent_ids(reach, key):
    """The non-zero up- or downstream reach ids SWORD records for a reach."""
    try:
        arr = np.atleast_2d(np.asarray(
            np.ma.filled(reach.metadata[key], 0), dtype='i8'))
        return [str(v) for v in np.unique(arr) if v > 0]
    except Exception:
        return []


def _apply_corridor_factors(extractor):
    """
    Multiply the per-node corridor coefficients, and report the resulting
    cross-channel half-corridor in metres so the effect is visible rather than
    implicit.

    Takes the whole extractor so the log lines can name the SWORD reach_id.
    They used to print reach.reach_index, which patch 2 has by then
    broadcast from a scalar to one value per node, so the line read
    "corridor reach [3 3 3 3 ...]".
    """
    if WTH_COEF_FACTOR == 1.0 and EXT_DIST_COEF_FACTOR == 1.0:
        return
    reaches = getattr(extractor, 'reach', None) or []
    ids = getattr(extractor, 'reach_idx', None) or []
    for r in reaches:
        for attr, factor in (('wth_coef', WTH_COEF_FACTOR),
                             ('ext_dist_coef', EXT_DIST_COEF_FACTOR)):
            v = getattr(r, attr, None)
            if v is None:
                continue
            setattr(r, attr, np.asarray(v, dtype='f8') * factor)

    # Report what this means on the ground, for the first few reaches.
    for pos, r in enumerate(reaches[:5]):
        try:
            width = np.ma.filled(np.asarray(r.width, dtype='f8'), np.nan)
            maxw = np.ma.filled(np.asarray(r.max_width, dtype='f8'), np.nan)
            wth = np.asarray(r.wth_coef, dtype='f8')
            ext = np.asarray(r.ext_dist_coef, dtype='f8')
            node_len = np.asarray(r.node_length, dtype='f8')
            gate1 = np.nanmedian(2.0 * width * wth / 3.0)
            gate2 = np.nanmedian(ext * np.maximum(
                node_len, np.fmax(maxw, width) * wth))
            LOGGER.info('shim: corridor reach %s -- prior width %.0f m, '
                        'half-corridor gate1 %.0f m, gate2 %.0f m '
                        '(binding: %.0f m)',
                        ids[pos] if pos < len(ids) else '?',
                        np.nanmedian(width), gate1, gate2,
                        min(gate1, gate2))
        except Exception:
            pass
    LOGGER.warning('shim: corridor relaxed -- wth_coef x%.2f, ext_dist_coef '
                   'x%.2f. Node assignment may pick up water from adjacent '
                   'reaches (cross-reach double counting).',
                   WTH_COEF_FACTOR, EXT_DIST_COEF_FACTOR)


def absent_variables():
    """Per-pixel variables the shim had to substitute for, this session."""
    return sorted(_ABSENT_REPORTED)


install()

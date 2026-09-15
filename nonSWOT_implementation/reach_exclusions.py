"""Parse a SWORD reach exclusion list.

Shared by riverobs_shim.py, run_calval2rivertile.py, 1.2_run_RiverObs.py,
1.3_true_node_polygons.py and 1.4_true_reach_polygons.py so that every
`--exclude-reaches` in the pipeline means the same thing. Standard library
only, so the post-processing scripts can import it without pulling in
RiverObs.

How to exclude reaches
----------------------------------
Either an inline list of reach ids:

    --exclude-reaches 81270500291
    --exclude-reaches 81270500291,81270500281
    --exclude-reaches 81270500291 81270500281

or the path to a .txt file holding them:

    --exclude-reaches exclude_YR_AOI.txt

In a file, one id per line is the readable form, but comma- or
space-separated ids on one line work too.

    # YR_AOI -- single-channel stem only
    81270500291 

-----------------------------------------------------------------------------
Script by:
Camryn Kluetmeier (camryn.kluetmeier@duke.edu)

Parts of this script were developed with assistance from Claude Code
(Anthropic) for debugging, documentation, and related editorial suggestions.

Last updated: 2026-09-15

"""

PIPELINE_VERSION = '1.0.0'


import os


__all__ = ['parse', 'to_env', 'from_env', 'describe', 'ExclusionError']

REACH_ID_DIGITS = 11
NODE_ID_DIGITS = 14


class ExclusionError(ValueError):
    """A malformed exclusion list. Message is meant to be shown verbatim."""


def _split(text):
    """Tokenize one line or one inline value: strip comments, split on , and
    whitespace."""
    text = text.split('#', 1)[0]
    return [tok for tok in text.replace(',', ' ').split() if tok]


def _check(tok, origin):
    if not tok.isdigit():
        raise ExclusionError(
            '{}: {!r} is not a reach id. Expected {} digits, or the path to '
            'a .txt file.'.format(origin, tok, REACH_ID_DIGITS))
    if len(tok) == NODE_ID_DIGITS:
        raise ExclusionError(
            '{}: {} is {} digits, which is a SWORD node_id, not a reach_id. '
            'The reach_id is its first {} digits: {}. Exclusion is '
            'reach-level -- excluding a reach removes all of its nodes, and '
            'there is no way to exclude individual nodes from a reach.'.format(
                origin, tok, NODE_ID_DIGITS, REACH_ID_DIGITS,
                tok[:REACH_ID_DIGITS]))
    return int(tok)


def _looks_like_path(value):
    """A path, not an id list.

    Deliberately not os.path.exists(): a mistyped filename must raise
    'no such file' rather than fall through to the id parser and produce
    'not a reach id', which sends you looking in the wrong place.
    """
    lowered = value.lower()
    if lowered.endswith(('.txt', '.csv', '.lst', '.list')):
        return True
    return os.sep in value or (os.altsep and os.altsep in value)


def _read_file(path, warn):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise ExclusionError(
            'exclusion file not found: {}'.format(path))
    ids = []
    with open(path) as fobj:
        for lineno, line in enumerate(fobj, 1):
            for tok in _split(line):
                ids.append(_check(tok, '{}:{}'.format(path, lineno)))
    if not ids:
        warn('exclusion file {} holds no reach ids (only comments or blank '
             'lines) -- nothing will be excluded'.format(path))
    return ids


def parse(values, warn=None):
    """
    Resolve one or more --exclude-reaches values to a sorted tuple of ints.

    `values` may be a single string or an iterable of strings; each is
    either an inline id list or a path to a .txt file. Returns () for
    None or an empty iterable, so callers can treat "no exclusions" and
    "an empty file" the same way.

    `warn` is called with one message per non-fatal oddity (an empty
    file, a duplicate, an id whose digit count is not 11). Defaults to
    printing.
    """
    if warn is None:
        def warn(msg):
            print('  warning: ' + msg)

    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        values = [values]

    ids, seen = [], set()
    for value in values:
        if not isinstance(value, str):
            value = str(value)
        value = value.strip()
        if not value:
            continue
        if _looks_like_path(value):
            found = _read_file(value, warn)
        else:
            found = [_check(tok, 'exclusion list') for tok in _split(value)]
        for rid in found:
            if rid in seen:
                warn('reach {} listed more than once -- counted once'.format(
                    rid))
                continue
            seen.add(rid)
            ids.append(rid)

    for rid in ids:
        if len(str(rid)) != REACH_ID_DIGITS:
            warn('reach {} is {} digits; SWORD reach ids are {}. Accepting it, '
                 'but check it against the prior database.'.format(
                     rid, len(str(rid)), REACH_ID_DIGITS))
    return tuple(sorted(ids))


def to_env(ids):
    """Canonical comma-separated form, for handing the list to a subprocess."""
    return ','.join(str(int(rid)) for rid in ids)


def from_env(name='RIVEROBS_EXCLUDE_REACHES', warn=None):
    """Read the list back out of the environment. () when unset or empty."""
    return parse(os.environ.get(name, ''), warn=warn)


def describe(ids, limit=8):
    """One-line summary for a log or a netCDF attribute."""
    if not ids:
        return 'none'
    shown = ', '.join(str(rid) for rid in ids[:limit])
    if len(ids) > limit:
        shown += ', ... (+{} more)'.format(len(ids) - limit)
    return '{} reach{}: {}'.format(
        len(ids), '' if len(ids) == 1 else 'es', shown)


def add_argument(parser, dest='exclude_reaches'):
    """
    Register --exclude-reaches on an argparse parser, so the flag reads
    identically in every script that takes it.
    """
    parser.add_argument(
        '--exclude-reaches', dest=dest, nargs='+', default=None,
        metavar='REACH_ID|FILE.txt',
        help='SWORD reach ids to leave out, as ids (space- or comma-'
             'separated) and/or the path to a .txt file with one id per '
             'line (# comments allowed). Repeatable; the union is used. Use '
             'this for a reach that SWORD runs parallel to the stem you are '
             'profiling.')
    return parser


if __name__ == '__main__':
    # Dry run: resolve the arguments and print what they mean, so an
    # exclusion file can be checked without starting a RiverObs job.
    import argparse
    p = argparse.ArgumentParser(
        description='Resolve an exclusion list and print it. Changes nothing.')
    add_argument(p)
    a = p.parse_args()
    print('# reach_exclusions.py {}'.format(PIPELINE_VERSION))
    try:
        resolved = parse(a.exclude_reaches)
    except ExclusionError as exc:
        raise SystemExit('error: {}'.format(exc))
    print('resolved: {}'.format(describe(resolved, limit=1000)))
    print('env form: RIVEROBS_EXCLUDE_REACHES={}'.format(to_env(resolved)))

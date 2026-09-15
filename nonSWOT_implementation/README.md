# 3.1 RiverObs orthomosaic width pipeline

**Deriving SWOT-comparable river width from digitized water masks, using the
SWOT RiverObs processor.**

Takes a water mask and processes it through RiverObs, the same software that
generates the operational SWOT River Single-Pass (RiverSP) vector product.

---

## Table of contents

1. [Pipeline overview](#1-pipeline-overview)
2. [Repository contents](#2-repository-contents)
3. [Installation](#3-installation)
4. [Patching RiverObs](#4-patching-riverobs)
5. [Input requirements](#5-input-requirements)
6. [Quick start](#6-quick-start)
7. [Step reference](#7-step-reference)
8. [Configuration reference](#8-configuration-reference)
9. [Adapting to a new study area](#9-adapting-to-a-new-study-area)
10. [Citation](#10-citation)

---

## 1. Pipeline overview

This pipeline converts a water mask into node- and
reach-level river widths on the SWORD prior river network, so that widths
measured from another source can be referenced against the SWOT
RiverSP product. Its products are two combined CSVs (one row per node, one row
per reach), node and reach footprint shapefiles recovered from RiverObs's own
pixel assignment, and a node QC table for nodes a manual review excluded. `3.2_ortho_node_width_diff.R` and `3.3_ortho_reach_width_diff.R` read
them.

RiverObs assigns each water cell to a prior node, sums the
assigned cell areas, and divides the sum by the SWORD prior `node_length`.

Only the width variables in the output are meaningful. Pixel-cloud heights are
constant zero and all WSE and slope fitting is disabled, so `wse`, `wse_u` and
`slope` have no useful information.

---

## 2. Repository contents

| File | Purpose |
|---|---|
| `apply_riverobs_patches.py` | Applies the four source edits to a RiverObs checkout. Idempotent, with `--status` and `--revert`. |
| `riverobs_shim.py` | Six runtime patches applied by import (four guards, the opt-in corridor relaxation, and the opt-in reach exclusion). |
| `reach_exclusions.py` | Parses `--exclude-reaches` (id list or `.txt` file) for every script that takes it. Standard library only. |
| `run_calval2rivertile.py` | Wrapper around RiverObs's `calval2rivertile.py` that loads the shim first. |
| `1.0_check_SWORD_nc.py` | Verifies a prior-database netCDF is usable, and that width variables are present. |
| `1.1_rasterize_watermask.py` | Burns a water-mask shapefile to a regular grid. |
| `1.2_run_RiverObs.py` | Runs every survey × prior-database version; outputs to CSV. |
| `1.3_true_node_polygons.py` | Recovers the actual per-node water footprints from the PIXCVec assignment. |
| `1.4_true_reach_polygons.py` | The same dissolve keyed on `reach_id`, with reach-level attributes, along-stream ordering and a SWORD bifurcation report. |
| `run_all.sh` | Batch driver for steps 1, 2, 3b and 3c. Site paths, `RES`, `VERSIONS`, `EXCLUDE_SHP` and the survey manifest are edited directly in the variable block at the top of the script. |
| `riverobs_ortho_v16.rdf`, `riverobs_ortho_v17b.rdf` | Annotated RiverObs configuration. The two differ only in the header comment and `reach_db_path`; edit that path to point at your local SWORD netCDF. |

---

## 3. Installation

### 3.1 Environment

Create an environment (named `RiverObs`) and install the packages the
scripts in this directory import:

- `numpy`
- `pandas`
- `geopandas`
- `rasterio`
- `shapely`
- `netCDF4`
- `pyproj` - used by the water-mask converter that
  `apply_riverobs_patches.py` inserts into RiverObs

RiverObs itself is not on PyPI or conda-forge; install it separately from its
own repository, as described in 3.2 below. Add two further packages that
RiverObs imports at the top of `SWOTRiverEstimator.py` but that are absent
from its own `Install.md` package list: **`bottleneck`** and
**`piecewise-regression`**.

### 3.2 RiverObs

RiverObs is used from source, not installed:

```bash
git clone https://github.com/SWOTAlgorithms/RiverObs.git
export PYTHONPATH="/path/to/RiverObs/src:$PYTHONPATH"
```

Add the `export` to your shell profile to make it persistent. Edits under
`src/` take effect on the next run with no rebuild.

Verify:

```bash
python -c "import SWOTRiver.Estimate; from RiverObs.ReachDatabase import ReachExtractor; print('ok')"
```

A `SyntaxWarning` from `calval.py` is expected and harmless — it is present
upstream.

### 3.3 Prior database

RiverObs reads the SWORD **netCDF** distribution. Point
`reach_db_path` at a continent-level `.nc` file or at a directory of them.

---

## 4. Patching RiverObs

```bash
python apply_riverobs_patches.py --riverobs-root /path/to/RiverObs
```

### 4.1 Source edits (`apply_riverobs_patches.py`)

| File | Edit | Rationale |
|---|---|---|
| `src/SWOTRiver/products/calval.py` | Implement the `from_airborne_imagery` stub | Upstream reserves this classmethod for exactly this input type but leaves it `NotImplementedError`. |
| `src/SWOTRiver/products/calval.py` | Declare `geolocation_qual`, `classification_qual`, `sig0_qual` on `SimplePixelCloud` | Two are commented out upstream and one is misspelled `classificaiton_qual`; `sig0_qual` is absent. Without all three, `process_node()` raises `TypeError: 'NoneType' object is not subscriptable`. |
| `src/bin/calval2rivertile.py` | Register the `airborne_watermask` format | Makes the converter reachable from the CLI. |
| `src/bin/calval2rivertile.py` | Add the dispatch branch | As above. |

### 4.2 Runtime patches (`riverobs_shim.py`)


| Patch | Problem without it |
|---|---|
| `SWOTRiverEstimator.__init__` — default `reach_pct_good_sus_thresh` to 0 | `TypeError: '<=' not supported between instances of 'float' and 'NoneType'`. `CalValToRiverTile` never passes this argument although `L2PixcToRiverTile` does. |
| `RiverReach.__init__` — broadcast scalar attributes to node count | `ValueError: zero-dimensional arrays cannot be concatenated`. Under `height_agg_method='orig'`, `wse_s_u` is a scalar constant. |
| `ReachExtractor.__init__` — pad `centerline_lon`/`centerline_lat` to equal length | `ValueError: setting an array element with a sequence ... inhomogeneous shape`. Prior-database reaches have different centerline point counts; since numpy 1.24 a ragged list raises. The product declares these as 2-D `(reaches, centerlines)`, so a rectangular array is required anyway, and `rivertile.py` already filters the padding via `np.abs(lats) < 90`. |
| `RiverObs.get_node_stat` — substitute for two absent variables | `AttributeError: 'RiverNode' object has no attribute 'time_from_prev_xover'`. Per-line crossover timing does not exist in a water mask. Only these two are substituted; every other missing variable re-raises so RiverObs's own handlers run unchanged. |
| `ReachExtractor.__init__` — remove excluded reaches from the prior database (opt-in) | Not a crash: SWORD represents some bifurcations as two parallel reaches, and an along-stream profile wants one of them. Set `RIVEROBS_EXCLUDE_REACHES` or pass `--exclude-reaches`. **Changes pixel assignment for the reaches that remain** — see [Reach exclusion](#reach-exclusion--single-channel-runs). |

---

## 5. Input requirements

### 5.1 Water mask

- Polygon shapefile (or any OGR-readable vector format)
- Projected CRS with meter units
- Water only — the pipeline treats every polygon as water

### 5.2 Prior database

SWORD netCDF for the region of interest, one file per version you intend to
compare against.

### 5.3 Survey manifest

`1.2_run_RiverObs.py` takes its surveys on the command line — `--name` (one or
more basenames, matching what step 1 was given) and an optional `--date` (the
SWOT overpass date, one per name or one for all). There is no hard-coded list
to edit; a one-off AOI is just

```bash
python 1.2_run_RiverObs.py ... --name YR_AOI --res 10
```

`run_all.sh` keeps a `SURVEYS` block (name | water-mask basename | orthomosaic
basename) because step 1 needs the mask and ortho paths as well, and it feeds
the names straight through to `--name`.

Versions past 2.0.0 of `1.2_run_RiverObs.py` no longer carry a `SURVEYS` list
or a `--survey` flag. If a script of yours passes `--survey`, change it to
`--name`.

### 5.4 Reach exclusion list (optional)

A plain text file of SWORD reach ids to leave out, one per line, `#` for
comments:

```
# YR_AOI -- single-channel stem only
81270500291   # side channel, bypasses 81270500031 / 81270500041
```

Pass it, or the ids themselves, to `--exclude-reaches`. See
[Reach exclusion](#reach-exclusion--single-channel-runs).

---

## 6. Quick start

```bash
conda activate RiverObs
export PYTHONPATH="/path/to/RiverObs/src:$PYTHONPATH"

# one-time
python apply_riverobs_patches.py --riverobs-root /path/to/RiverObs
python 1.0_check_SWORD_nc.py /path/to/na_sword_v17b.nc --verify
$EDITOR run_all.sh              # paths, RES, VERSIONS, EXCLUDE_SHP,
                                # EXCLUDE_REACHES, SURVEYS
$EDITOR riverobs_ortho_v17b.rdf # reach_db_path; likewise riverobs_ortho_v16.rdf

# batch: steps 1, 2 and 3b produce the node polygons; nothing is excluded yet
bash run_all.sh 1 2 3b
bash run_all.sh 1               # a single step

# manual QGIS review: digitize cloud / ragged edges into ONE exclusion layer
# covering every survey, at the EXCLUDE_SHP path set in run_all.sh

# step 3c applies it to every survey and version at once
bash run_all.sh 3c              # re-run this alone after any edit to the layer

# or the whole thing, once the exclusion layer exists
bash run_all.sh

# step 4 builds the reach footprints, and reports where SWORD bifurcates
bash run_all.sh 4

# single-channel: pick a branch from that report, put it in EXCLUDE_REACHES,
# and re-run. See "Reach exclusion" below for which end to exclude it at.
```

---

## 7. Step reference

### Step 0 — prior-database check

```bash
python 1.0_check_SWORD_nc.py /path/na_sword_v17b.nc --verify
```

Read-only. Reports how many of RiverObs's declared variables the file carries,
how many of the width-critical subset are present, and whether the required
`/reaches` subgroups exist. `--verify` loads the database through RiverObs and
calls one reach, which exercises the full read path.

**Missing variables are normal and harmless.** RiverObs declares far more than
the public releases carry; `Product.__getattr__` returns a fully-masked array
for anything absent. `/reaches/type` is missing from public SWORD and RiverObs
derives it from `reach_id % 10`, which is exact.

The check matters precisely *because* absence is silent rather than fatal: a
database missing `node_length` would not crash, it would produce plausible and
wrong widths. `node_length` is the divisor in `width = area / node_length`;
`max_width` and `ext_dist_coef` size the search corridor.

**Exit codes:** 0 usable, 1 not usable.

---

### Step 1 — rasterize the water mask

```bash
python 1.1_rasterize_watermask.py \
    /path/water_masks/<survey>_watermask.shp \
    /path/output/rasterized_water_masks \
    --name <survey> --res 3 \
    [--ortho-tif /path/orthomosaics/<survey>_ortho.tif]
```

Burns every polygon feature to a `uint8` grid, snapped so cell area is exactly
`res²`. `all_touched=False` — a cell must have its centre inside the polygon —
which keeps the area estimate unbiased.

**Output:** `<name>_water_<res>m.tif` (the only file step 2 needs).

**Grid resolution.** Node area is a sum of cell areas, so spacing sets only the
quantization of that sum. At 3 m a 200 m node in a 50 m channel holds ~3,300
cells and quantization error is a fraction of a percent — far below digitization
and centerline error. Going finer multiplies pixel-cloud size quadratically and
forces `SWOTRiverEstimator`'s connected-component segmentation to allocate an
image the size of the full survey grid. **3–5 m is the recommended range.**
Measured accuracy at 3 m against a channel of known area: **0.025%**.

`--pad` (default 2000 m) must exceed the largest prior `max_width` in the area,
or the search corridor runs off the edge of the grid.

`--ortho-tif` additionally writes a footprint raster and reports water cells
falling outside the imaged area — a useful one-off check that digitizing stayed
within the survey. Nothing downstream reads it.

---

### Step 2 — run RiverObs

```bash
python 1.2_run_RiverObs.py \
    --riverobs-root /path/to/RiverObs \
    --raster-dir  /path/output/rasterized_water_masks \
    --rdf-dir     . \
    --out-dir     /path/output/riverobs \
    --res 3 --versions v17b v16 --keep-going
```

Loops over the `SURVEYS` list × `--versions`, invoking `run_calval2rivertile.py`
for each pair, then flattens the RiverTile node and reach groups into two
combined CSVs.

| Option | Effect |
|---|---|
| `--name NAME [NAME ...]` | The water-mask basenames to process (required) |
| `--date YYYY-MM-DD [...]` | SWOT overpass date per name, or one for all. Written to the `SWOT_date` column; left empty if omitted |
| `--exclude-reaches ID\|FILE.txt` | SWORD reaches to drop from the prior database before assignment. See [Reach exclusion](#reach-exclusion--single-channel-runs) |
| `--out-tag TAG` | Suffix for the combined CSV names, so an excluded run does not overwrite the full one |
| `--keep-going` | Continue past a failed survey, still write CSVs for the rest, list failures and exit non-zero |
| `--log-level` | `debug` / `info` / `warning` / `error` |
| `--wth-coef-factor` | Multiply the prior `wth_coef`, widening both search-corridor gates. `1.0` is stock RiverObs. Overrides `RIVEROBS_WTH_COEF_FACTOR` |
| `--ext-dist-coef-factor` | Multiply the prior `ext_dist_coef` (gate 2). `1.0` is stock RiverObs. Overrides `RIVEROBS_EXT_DIST_COEF_FACTOR` |

**Outputs**

```
<out-dir>/<version>/ortho_<survey>_<version>.nc            RiverTile
<out-dir>/<version>/ortho_<survey>_<version>_pixcvec.nc    per-pixel node assignment
<out-dir>/<version>/ortho_<survey>_<version>_pixc.nc       reformatted pixel cloud
<out-dir>/ortho_riverobs_nodes_all.csv
<out-dir>/ortho_riverobs_reaches_all.csv
```

**Expected log noise.** These are normal and can be ignored: `PIXC Quality flags
not found in input file`; `converting a masked element to nan`; `divide by zero
encountered in divide` at `ww = 1/(wse_r_u**2)`; `Mean of empty slice`;
`assign_reaches: no observations mapped to nodes in this reach` for reaches
outside the mask; and the shim's `time_from_prev_xover` / `time_to_next_xover`
substitution notice. All originate in the deliberately disabled WSE machinery
(see [§8](#8-configuration-reference)).

---

### Reach exclusion — single-channel runs

SWORD represents some bifurcations as two parallel reaches over two channels.
For an along-stream profile tied to SWORD, only one of them is the stem: the
other contributes a second value at the same `p_dist_out`, and its nodes reach
across the floodplain into water the stem does not own.

`1.4_true_reach_polygons.py --rivertile ...` ends by listing every place the
prior network branches, straight from the product's own `n_reach_up` /
`n_reach_dn`, and names the branches. That is the list to choose from. Which
branch is the stem is a judgement about the river, so look at the pair in QGIS
first.

Both ends of the pipeline take the same flag, and they are **not**
interchangeable.

| Where | What happens | Use it when |
|---|---|---|
| `1.2_run_RiverObs.py` / `run_calval2rivertile.py` | The reach is removed from the prior database before RiverObs runs. It appears in no RiverTile, PIXCVec or CSV. **Its water is freed for neighbouring reaches.** | The reach should not exist for this project at all |
| `1.3_true_node_polygons.py` / `1.4_true_reach_polygons.py` | Its pixels are dropped from a finished PIXCVec. Assignment is not re-done, so every kept footprint is unchanged. | You want the parallel channel out of the output but the stem exactly as the run computed it |

**Why excluding upstream moves water.**
`SWOTRiverEstimator.assign_reaches` makes the assignment competitive and
unique. It builds one `Centerline` from every reach's nodes, takes each pixel's
distance to that tile-wide centerline as the target to beat

```python
min_dist = tile_d
...
mask = river_obs.d <= these_min_dists
```

and lets a reach keep a pixel only when it does better — so each water cell
ends up with the single nearest reach. Remove a reach from the prior and its
channel's water is not discarded; it is simply available to whatever kept reach
is within corridor range. The neighbour's `width`, `area_total` and node
footprints then grow across the bifurcation.

Patch 6 measures this instead of leaving it implicit. For each excluded reach
it reports the nearest kept reaches, the distance in metres, and the
half-corridor from the patch-4 gates:

```
shim: excluding reach 81270500291 (76 nodes, SWORD up 81270500051 / dn 81270500021)
shim:   nearest kept reach 81270500041 -- 412 m away, half-corridor 3333 m -> WILL LIKELY ABSORB IT
```

A kept reach closer than its own half-corridor will take the excluded channel.
At `--wth-coef-factor 5 --ext-dist-coef-factor 5` on a braided river that is
the expected case, not the exception — which is why the post-hoc route exists.

**Usage**

```bash
# prior-database exclusion: re-runs RiverObs without the reach
python 1.2_run_RiverObs.py \
    --riverobs-root /path/to/RiverObs \
    --raster-dir /path/GIS --rdf-dir . --out-dir /path/GIS/RiverObs \
    --name YR_AOI --res 10 --versions v17b \
    --wth-coef-factor 5.0 --ext-dist-coef-factor 5.0 \
    --exclude-reaches 81270500291 \
    --out-tag single_channel

# post-hoc exclusion: seconds, on the PIXCVec you already have
python 1.4_true_reach_polygons.py \
    /path/GIS/YR_AOI_water_10m.tif \
    /path/GIS/RiverObs/v17b/ortho_YR_AOI_v17b_pixcvec.nc \
    /path/GIS/RiverObs/true_reach_polygons_YR_AOI_single_channel.shp \
    --rivertile /path/GIS/RiverObs/v17b/ortho_YR_AOI_v17b.nc \
    --exclude-reaches 81270500291
```

An id outside the bounding box is reported and ignored rather than treated as
an error — one exclusion file is meant to be reusable across AOIs. Check the
report if you expected a reach to go. An exclusion list covering *every* reach
in the tile is refused, because an empty prior database fails much later with a
far less obvious message.

Both routes record themselves. `run_calval2rivertile.py` stamps
`riverobs_excluded_reaches_requested`, `riverobs_excluded_reaches` (what was
actually dropped), `riverobs_excluded_reaches_not_in_tile` and
`riverobs_exclusion_mode` on the RiverTile and PIXCVec as global attributes,
and steps 1.3 / 1.4 print them back, so a finished product says which list
produced it.

> **Reading coverage after an exclusion.** Step 1.3 / 1.4's "mask cells
> assigned to a node" percentage drops when reaches are excluded, because the
> excluded channel's water is genuinely not in the output. The scripts say so
> when they know an exclusion is in play — do not widen the corridor in
> response to a gap you created on purpose.

> **`rch_id_up` / `rch_id_dn` still name excluded reaches.** SWORD topology is
> copied through from the prior database, so a kept reach next to an excluded
> one still lists it as a neighbour. RiverObs handles the dangling reference
> (`except IndexError: continue`), and the attribute is left alone rather than
> rewritten, so the product still shows the real prior network.

---

### Step 3b — node footprints

```bash
python 1.3_true_node_polygons.py \
    /path/rasterized_water_masks/<survey>_water_3m.tif \
    /path/riverobs/<version>/ortho_<survey>_<version>_pixcvec.nc \
    /path/node_qc/true_node_polygons_<survey>_<version>.shp \
    --rivertile /path/riverobs/<version>/ortho_<survey>_<version>.nc \
    [--reaches]
```

The PIXCVec output carries a `node_id` for every water cell. Dissolving by it
recovers the exact region each node's area and width were summed over. Verified
against the RiverTile: polygon areas match `area_total` to **0.000 m²**, and
`polygon area / p_length` reproduces `width` exactly.

`--rivertile` joins `width`, `p_length`, `p_dist_out` and `n_good_pix` so the
layer can be styled by width. The script also prints nodes narrower than 60% of
the median as a **hint** for the review — nothing is excluded.

**Attributes written:** `node_id`, `node_str` (string copy, authoritative if the
numeric field is ever mangled by a DBF), `n_cells`, `area_m2`, `keep`
(initialized to 1), plus the RiverTile join if requested (`reach_id`, `width`,
`p_length`, `p_dist_out`, `n_good_pix`).

`--exclude-reaches` drops named reaches from the output without re-running
anything; see [Reach exclusion](#reach-exclusion--single-channel-runs).
`--reaches` writes a bare reach-level dissolve alongside — [Step 4](#step-4--reach-footprints)
does the same thing with reach attributes and along-stream ordering attached
and is the better choice.

---

### Step 3c — exclusion list

**Route A — exclusion polygon (recommended).** Digitize a polygon layer covering
the areas you do not trust. **One layer holding polygons for every survey is
fine** — the test is spatial, so a polygon only affects nodes it overlaps.

Batch, every survey and every version in one call (this is what `run_all.sh 3c`
runs):

```bash
python 3.1.4_manual_node_qc.py --batch \
    --nodes-dir   node_qc \
    --exclude-shp digitizing/bad_ends_watermasks.shp \
    --versions v17b v16 \
    --out-dir node_qc \
    --out node_qc/node_qc_all.csv
```

It discovers every `true_node_polygons_<survey>_<version>.shp` under
`--nodes-dir`, writes one CSV per survey/version plus the merged `--out`, and
prints a kept/dropped table. A survey that fails is reported and skipped rather
than taking the rest of the run down.

Single survey/version:

```bash
python 3.1.4_manual_node_qc.py \
    --nodes  node_qc/true_node_polygons_<survey>_<version>.shp \
    --exclude-shp digitizing/bad_ends_watermasks.shp \
    --survey <survey> --sword-version <version> \
    --out node_qc/node_qc_<survey>_<version>.csv
```

Nodes overlapping the exclusion area by more than `--max-overlap` (default 5%)
are dropped; near-misses are listed. Because the exclusion is spatial, one act
of judgement filters **every** prior-database version consistently — different
versions place nodes differently, but the cloud is in the same place on the
ground. It is also the reproducible artefact: a reader sees which ground area
was excluded rather than an unexplained list of identifiers.

> **If two surveys overlap on the ground.** The whole layer is applied to every
> survey. That is right when the polygons mark ground that is bad in all of
> them, and wrong when two surveys cover the same reach on different dates with
> different bad areas — a polygon drawn for one date would also clip the other.
> In that case add a text field naming the survey each polygon belongs to and
> pass `--exclude-field <name>`. Polygons whose field is empty or `all` still
> apply everywhere. The script prints the layer's field names on startup.

**Route B — `keep` field.** Set `keep = 0` in QGIS and omit `--exclude-shp`.
Deleting rows works too, with `--original` pointing at an untouched copy.
`--exclude-ids` accepts a comma-separated list. This route is per-layer by
nature and has no batch form.

**Merge existing CSVs:**

```bash
python 3.1.4_manual_node_qc.py --merge 'node_qc/node_qc_*.csv' \
    --out node_qc/node_qc_all.csv
```

Step 3c reads nothing but the step 3b shapefiles and the exclusion layer, so
after any edit to that layer in QGIS, `bash run_all.sh 3c` alone refreshes
`node_qc_all.csv` in seconds — nothing upstream is repeated.

---

### Step 4 — reach footprints

```bash
python 1.4_true_reach_polygons.py \
    /path/rasterized_water_masks/<survey>_water_<res>m.tif \
    /path/riverobs/<version>/ortho_<survey>_<version>_pixcvec.nc \
    /path/out/true_reach_polygons_<survey>_<version>.shp \
    --rivertile /path/riverobs/<version>/ortho_<survey>_<version>.nc \
    [--exclude-reaches ID|FILE.txt] [--min-obs-frac 0.5]
```

The reach-level counterpart of step 3b: the same dissolve, keyed on the
`reach_id` RiverObs wrote into the PIXCVec. One polygon per SWORD reach,
holding the water that reach actually won.

This is the sampling unit for a reach-level along-stream profile of anything on
a grid — Sentinel-2 ice cover, SWOT raster WSE and its variability. `ds_rank`
is the along-stream order already computed from `p_dist_out`: 1 is the most
downstream observed reach.

Geometry is the true observed footprint, so islands come through as holes and a
reach split around a bar comes out multipart. That is deliberate: a filled or
convex hull would pull non-water cells into the zonal statistics. `n_parts`
reports the multipart count, and the run lists the most fragmented reaches.

`--rivertile` is strongly recommended. It attaches the reach-level fields and
enables the bifurcation report — the list of reaches to consider excluding.
Each branch is classified from SWORD's own labelling: a branch appearing at
*two* branching reaches leaves the network at one and rejoins at the other,
which is a side channel around a stem and the unambiguous candidate; a branch
appearing once is a separate arm, and its `river_name` relative to the reach it
branches from says whether it is a split of the same river or a tributary. On
the YR_AOI tile this identifies `81270500291` — it branches off `81270500021`
and rejoins at `81270500051`, bypassing `81270500031` / `81270500041`.

`--min-obs-frac FRAC` additionally sets `keep = 0` on reaches observed on fewer
than FRAC of their SWORD nodes. Nothing is deleted; the polygons stay in the
file so you can see what was flagged. A reach observed on a third of its nodes
has a mean over a third of its length, which is not comparable to a fully
observed neighbour on the same profile.

**Attributes written:** `reach_id`, `reach_str` (string copy), `n_cells`,
`area_m2`, `keep`, `n_nodes_ob` (distinct nodes that received water — not the
same as the RiverTile's quality-gated `n_good_nod`), `n_parts`, plus, with
`--rivertile`: `width`, `width_u`, `area_total`, `area_detct`, `obs_frac_n`,
`partial_f`, `n_good_nod`, `p_length`, `p_width`, `p_dist_out`, `p_n_nodes`,
`river_name` and `ds_rank`.

`wse` and `slope` are deliberately absent — pixel-cloud heights are constant
zero in this pipeline, so they carry no information. See
[§8](#8-configuration-reference).

---

## 8. Configuration reference

The RiverObs `.rdf` passed to `calval2rivertile.py`. Values not listed take
RiverObs defaults. Keep every parameter except `reach_db_path` identical across
prior-database versions, or a cross-version comparison measures your settings
rather than the products.

| Key | Value | Why |
|---|---|---|
| `reach_db_path` | path to SWORD `.nc` | The only line that changes between versions. |
| `class_list` | `[4]` | Must equal `class_value` in `from_airborne_imagery`. |
| `use_fractional_inundation` | `[False]` | A digitized mask is binary. |
| `use_segmentation` | `[True]` | **Do not set False.** `SWOTRiverEstimator` always passes a `max_width`, and `flag_out_channel_and_label()` calls `.copy()` on the labels unconditionally, so `False` raises `AttributeError`. Because the converter sets `range_index`/`azimuth_index` to raster column/row, segmentation is a proper connected-component labelling on the mask grid. |
| `area_agg_method` | `orig` | Node area = `sum(pixel_area)`, width = area / `p_length`. The `composite` method is built around SWOT's water fraction, sig0 and edge classes, which a digitized mask does not have. |
| `height_agg_method` | `orig` | **Must also be `orig`.** RiverObs skips `get_node_agg()` only when height *and* area are both `orig`; any other value enters the interferometric height aggregator, which needs `phase_noise_std` / `dh_dphi`. |
| `ds` | `None` | Keeps the prior node spacing, so output `node_id` matches the SWOT product one-to-one. |
| `minobs` | `10` | Permissive by design — report thin coverage and filter it explicitly in the review. |
| `min_fit_points` | `1000000` | Set above any real node count so every WSE/slope fit is skipped. Heights are constant zero, so `wse_r_u` is 0 and the WLS weights `1/wse_r_u²` are infinite (`LinAlgError: SVD did not converge`). No fit touches area or width — reach width uses `mask_area`, not `mask_wse`. |
| `outlier_method` | `None` | Nothing for outlier rejection to work on. |
| `scalar_max_width` | `600.0` | Fallback only; RiverObs scales the corridor per node from prior `max_width` and `ext_dist_coef`. |
| `reach_pct_good_sus_thresh` | `0` | Every mask cell is good, so the tested percentage is always 100 and any lower value is neutral. |
| quality thresholds | `0` | A digitized mask carries no per-pixel quality information. |

> **`wse`, `wse_u` and `slope` in the output are meaningless.** Pixel-cloud
> heights are zero and all WSE fitting is disabled. Read only `width`,
> `area_total`, `area_detct`, `p_length`, `n_good_pix`, and the reach-level
> `obs_frac_n` / `partial_f`.

---

## 9. Adapting to a new study area

1. **`run_all.sh`** — the variable block at the top: paths, `RES`,
   `VERSIONS`, `EXCLUDE_SHP`, and the `SURVEYS` manifest.
2. **`1.2_run_RiverObs.py`** — `SWORD_VERSIONS` only (version tag → `.rdf`
   filename). Surveys are passed as `--name` / `--date`, not edited into the
   script.
3. **`.rdf`** — `reach_db_path` for your region.
4. **CRS** — `--crs` on step 1, default `EPSG:32606`. Any projected CRS
   with meter units works; the converter reads the CRS from the raster.
5. **`--pad`** — must exceed the largest prior `max_width` in the area.
6. **`--res`** — 3–5 m for channels of tens to hundreds of metres. Scale up for
   very wide rivers; verify against a polygon of known area either way.

---

## 10. Citation

- RiverObs — https://github.com/SWOTAlgorithms/RiverObs
- Altenau, E. H., et al. (2021), The Surface Water and Ocean Topography (SWOT)
  Mission River Database (SWORD): A global river network for satellite data
  products, *Water Resources Research*, 57, e2021WR030054.

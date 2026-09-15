#!/usr/bin/env bash
# =============================================================================
# run_all.sh -- steps 1, 2, 3b, 3c and 4 for every water mask
# -----------------------------------------------------------------------------
#   conda activate RiverObs
#   cd .../nonSWOT_implementation
#   bash run_all.sh                  # everything
#   bash run_all.sh 1                # step 1 only
#   bash run_all.sh 2 3b             # steps 2 and 3b only
#   bash run_all.sh 3c               # just re-apply the node exclusion layer
#   bash run_all.sh 4                # reach footprints + bifurcation report
#
# -----------------------------------------------------------------------------
# Script by:
# Camryn Kluetmeier (camryn.kluetmeier@duke.edu)
#
# Parts of this script were developed with assistance from Claude Code
# (Anthropic) for debugging, documentation, and related editorial suggestions.
#
# Last updated: 2026-09-15
#
# =============================================================================

set -uo pipefail

# Paths
BASE=/Users/camryn/Documents/UNC/_Tier1_sites/expanded_Yukon_Flats
SCRIPTS=/Users/camryn/Documents/UNC/_Tier1_sites/_data_management/YR2024_scripts/SWOTCalVal_Yukon24/3_Orthomosaic_SWOT_harmonization/3.1_RiverObs
RIVEROBS=/Users/camryn/Documents/UNC/_Tier1_sites/_data_management/YR2024_scripts/RiverObs

MASK_DIR=$BASE/Orthomosaics/width_validation
ORTHO_DIR=$BASE/Orthomosaics/orthos_resampled_3m/gap_filled
RASTER_DIR=$BASE/Orthomosaics/width_validation/rasterized_water_masks
OUT_DIR=$BASE/Orthomosaics/width_validation/RiverObs_output
QC_DIR=$BASE/Orthomosaics/width_validation/RiverObs_output/node_qc

# Hand-digitized polygons over cloud, haze and ragged survey edges. One layer
# covering all six orthos. The test is spatial, so it is valid for v16 and v17b
# alike -- node ids differ between SWORD versions, the ground does not.
EXCLUDE_SHP=$BASE/Orthomosaics/width_validation/digitizing/bad_ends_watermasks.shp

# If a polygon in EXCLUDE_SHP should apply to only ONE survey (two surveys
# covering the same reach on different dates, with different bad areas), add a
# text field to the layer naming the survey and set EXCLUDE_FIELD to its name.
# Leave empty to apply every polygon to every survey.
EXCLUDE_FIELD=""

# Fraction of a node polygon inside the exclusion area that is tolerated
# before the node is dropped.
MAX_OVERLAP=0.05

# -----------------------------------------------------------------------------
# Reach exclusion -- parallel channels at SWORD bifurcations
# -----------------------------------------------------------------------------
# SWORD represents some bifurcations as two parallel reaches over two channels.
# For an along-stream profile only one of them is the stem. Name the ones to
# leave out here, as ids or as a path to a .txt file with one id per line:
#
#   EXCLUDE_REACHES="81270500291"
#   EXCLUDE_REACHES="exclude_${SITE}.txt"
#
# Leave empty for no exclusion. Step 4 reports where SWORD bifurcates and names
# the branch reaches, so run it once first and pick from that list.
#
# WHERE this is applied matters, and the two are not interchangeable:
#
#   EXCLUDE_AT=prior     step 2 removes the reach from the prior database
#                        before RiverObs runs. Nothing downstream mentions it.
#                        But assignment is competitive and unique, so its
#                        channel's water is freed for neighbouring reaches and
#                        THEIR widths and footprints grow across the
#                        bifurcation. The shim reports the nearest kept reach
#                        and its half-corridor so you can see whether that
#                        happened.
#
#   EXCLUDE_AT=output    steps 3b and 4 drop the reach's pixels from the
#                        finished PIXCVec. Assignment is not re-done, so every
#                        kept footprint is exactly what the run computed.
#                        Costs seconds, and does not need step 2 re-run.
#
#   EXCLUDE_AT=both      applies it at both ends.
#
# See "Reach exclusion" in README.md for the argument.
EXCLUDE_REACHES=""
EXCLUDE_AT=prior

# Suffix for the combined CSVs and the step 4 shapefiles, so an excluded run
# does not overwrite the full one. Leave empty for the default names.
OUT_TAG=""

RES=3
VERSIONS="v17b v16"
LOG_LEVEL=info

# Search-corridor relaxation
# RiverObs sizes its cross-channel search corridor from the PRIOR channel
# width, so water lying well outside the prior channel -- anabranches, braidplain
# threads, wide side channels -- is excluded even when it is unambiguously river.
# These factors widen the corridor. 1.0 is stock RiverObs.
#
# Raise the factors until the fraction of the water mask assigned to a node
# plateaus near 100%, then stop; step 3b reports that fraction per survey. A
# wider corridor than you need lets adjacent reaches claim the same water --
# and makes EXCLUDE_AT=prior much more likely to move water to a neighbour.
export RIVEROBS_WTH_COEF_FACTOR=8.0
export RIVEROBS_EXT_DIST_COEF_FACTOR=8.0

STEPS="${*:-1 2 3b 3c 4}"
want () { case " $STEPS " in *" $1 "*) return 0;; *) return 1;; esac; }

# Build the flag lists once, so every step below agrees on them.
EXCL_PRIOR_ARG=""
EXCL_OUTPUT_ARG=""
if [ -n "$EXCLUDE_REACHES" ]; then
  case "$EXCLUDE_AT" in
    prior)  EXCL_PRIOR_ARG="--exclude-reaches $EXCLUDE_REACHES" ;;
    output) EXCL_OUTPUT_ARG="--exclude-reaches $EXCLUDE_REACHES" ;;
    both)   EXCL_PRIOR_ARG="--exclude-reaches $EXCLUDE_REACHES"
            EXCL_OUTPUT_ARG="--exclude-reaches $EXCLUDE_REACHES" ;;
    *)      echo "EXCLUDE_AT must be prior, output or both (got '$EXCLUDE_AT')"
            exit 1 ;;
  esac
  # Fail early on a malformed list rather than after step 2 has run.
  python reach_exclusions.py --exclude-reaches $EXCLUDE_REACHES || exit 1
fi

TAG_ARG=""
TAG_SUFFIX=""
if [ -n "$OUT_TAG" ]; then
  TAG_ARG="--out-tag $OUT_TAG"
  TAG_SUFFIX="_$OUT_TAG"
fi

mkdir -p "$RASTER_DIR" "$OUT_DIR" "$QC_DIR"
cd "$SCRIPTS" || exit 1

FAILED=""

# =============================================================================
# STEP 1 -- rasterize each water mask
# =============================================================================
if want 1; then
echo
echo "################ STEP 1: rasterize water masks ################"
echo "$SURVEYS" | while IFS='|' read -r NAME MASK ORTHO; do
  [ -z "$NAME" ] && continue
  echo
  echo "--- $NAME ---"
  python 1.1_rasterize_watermask.py \
      "$MASK_DIR/$MASK.shp" \
      "$RASTER_DIR" \
      --name "$NAME" \
      --res "$RES" \
      --ortho-tif "$ORTHO_DIR/$ORTHO.tif" \
    || echo "!!! step1 FAILED for $NAME"
done
fi

# =============================================================================
# STEP 2 -- run RiverObs
# =============================================================================
if want 2; then
echo
echo "################ STEP 2: RiverObs ################"
NAMES=$(echo "$SURVEYS" | awk -F'|' 'NF{printf "%s ", $1}')
python 1.2_run_RiverObs.py \
    --riverobs-root "$RIVEROBS" \
    --raster-dir "$RASTER_DIR" \
    --rdf-dir . \
    --out-dir "$OUT_DIR" \
    --name $NAMES \
    --res "$RES" \
    --versions $VERSIONS \
    --keep-going \
    --wth-coef-factor "$RIVEROBS_WTH_COEF_FACTOR" \
    --ext-dist-coef-factor "$RIVEROBS_EXT_DIST_COEF_FACTOR" \
    --log-level "$LOG_LEVEL" \
    $EXCL_PRIOR_ARG $TAG_ARG
STEP2_RC=$?
[ $STEP2_RC -ne 0 ] && FAILED="$FAILED step2(rc=$STEP2_RC)"
fi

# =============================================================================
# STEP 3b -- node polygons for review
# =============================================================================
if want 3b; then
echo
echo "################ STEP 3b: node polygons ################"
for V in $VERSIONS; do
  echo "$SURVEYS" | while IFS='|' read -r NAME MASK ORTHO; do
    [ -z "$NAME" ] && continue
    WATER="$RASTER_DIR/${NAME}_water_${RES}m.tif"
    PIXCVEC="$OUT_DIR/$V/ortho_${NAME}_${V}_pixcvec.nc"
    RIVERTILE="$OUT_DIR/$V/ortho_${NAME}_${V}.nc"
    OUT_SHP="$QC_DIR/true_node_polygons_${NAME}_${V}${TAG_SUFFIX}.shp"

    if [ ! -f "$RIVERTILE" ]; then
      echo "--- $NAME / $V : SKIPPED, no RiverTile (step 2 failed for this one)"
      continue
    fi
    echo
    echo "--- $NAME / $V ---"
    python 1.3_true_node_polygons.py \
        "$WATER" "$PIXCVEC" "$OUT_SHP" \
        --rivertile "$RIVERTILE" \
        $EXCL_OUTPUT_ARG \
      || echo "!!! step3b FAILED for $NAME / $V"
  done
done
fi

# =============================================================================
# STEP 4 -- reach polygons, and the SWORD bifurcation report
# =============================================================================
if want 4; then
echo
echo "################ STEP 4: reach polygons ################"
for V in $VERSIONS; do
  echo "$SURVEYS" | while IFS='|' read -r NAME MASK ORTHO; do
    [ -z "$NAME" ] && continue
    WATER="$RASTER_DIR/${NAME}_water_${RES}m.tif"
    PIXCVEC="$OUT_DIR/$V/ortho_${NAME}_${V}_pixcvec.nc"
    RIVERTILE="$OUT_DIR/$V/ortho_${NAME}_${V}.nc"
    OUT_SHP="$QC_DIR/true_reach_polygons_${NAME}_${V}${TAG_SUFFIX}.shp"

    if [ ! -f "$RIVERTILE" ]; then
      echo "--- $NAME / $V : SKIPPED, no RiverTile (step 2 failed for this one)"
      continue
    fi
    echo
    echo "--- $NAME / $V ---"
    python 1.4_true_reach_polygons.py \
        "$WATER" "$PIXCVEC" "$OUT_SHP" \
        --rivertile "$RIVERTILE" \
        $EXCL_OUTPUT_ARG \
      || echo "!!! step4 FAILED for $NAME / $V"
  done
done
fi

# =============================================================================
echo
echo "############################################################"
if [ -n "$FAILED" ]; then
  echo "Finished with problems:$FAILED"
  echo "Scroll up for '!!!' and 'FAILED' lines."
else
  echo "Finished."
fi
echo
echo "Outputs:"
echo "  water rasters   $RASTER_DIR"
echo "  RiverObs        $OUT_DIR"
echo "     combined CSVs: ortho_riverobs_nodes${TAG_SUFFIX:-_all}.csv,"
echo "                    ortho_riverobs_reaches${TAG_SUFFIX:-_all}.csv"
echo "  node polygons   $QC_DIR/true_node_polygons_<survey>_<version>${TAG_SUFFIX}.shp"
echo "  reach polygons  $QC_DIR/true_reach_polygons_<survey>_<version>${TAG_SUFFIX}.shp"
echo "  node QC         $QC_DIR/node_qc_all.csv   <- join downstream analysis on this"
echo
echo "Corridor factors in effect: wth_coef x$RIVEROBS_WTH_COEF_FACTOR,"
echo "                            ext_dist_coef x$RIVEROBS_EXT_DIST_COEF_FACTOR"
if [ -n "$EXCLUDE_REACHES" ]; then
  echo "Reach exclusion:            $EXCLUDE_REACHES  (at: $EXCLUDE_AT)"
else
  echo "Reach exclusion:            none -- every SWORD reach in the mask is kept"
fi
echo "Step 3b reports, per survey, how much of each water mask reached a node."
echo "Step 4 reports where SWORD bifurcates -- the candidates for EXCLUDE_REACHES."
echo
echo "Edited the node exclusion layer in QGIS? Re-run step 3c alone:"
echo "  bash run_all.sh 3c"
echo "Changed EXCLUDE_REACHES with EXCLUDE_AT=output? Re-run 3b and 4 only:"
echo "  bash run_all.sh 3b 4"
echo "############################################################"

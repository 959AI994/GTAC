#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./reset_epfl_act_out.sh [--src DIR] [--dst DIR] [--dry-run] [--yes]

Description:
  For each immediate subdirectory x under $dst (i.e. $dst/x):
    1) Clear everything under $dst/x/ACT_out
    2) Clear everything under $dst/x/merge_out_binary
    3) Remove $dst/x/merge_binary.log if it exists
    4) Copy all contents from $src/x into $dst/x/ACT_out

Safety:
  - No destructive actions by default; pass --yes to actually clear/delete.
  - Without --dry-run, changes are applied for real when --yes is set.

Default paths (relative to this script’s directory):
  ./EPFL
  ./graph_merge/EPFL
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR_DEFAULT="$REPO_ROOT/EPFL"
DST_DIR_DEFAULT="$REPO_ROOT/graph_merge/EPFL"

SRC_DIR="$SRC_DIR_DEFAULT"
DST_DIR="$DST_DIR_DEFAULT"
DRY_RUN=0
CONFIRM=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src)
      SRC_DIR="${2:-}"
      shift 2
      ;;
    --dst)
      DST_DIR="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --yes)
      CONFIRM=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "$SRC_DIR" ]]; then
  echo "Error: source directory does not exist: $SRC_DIR"
  exit 1
fi
if [[ ! -d "$DST_DIR" ]]; then
  echo "Error: destination directory does not exist: $DST_DIR"
  exit 1
fi

if [[ "$SRC_DIR" == "$DST_DIR" ]]; then
  echo "Error: src and dst must differ (avoid deleting source data by mistake)."
  exit 1
fi

run_cmd() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[dry-run] '
    printf '%q ' "$@"
    echo
  else
    "$@"
  fi
}

clear_dir_contents() {
  local d="$1"
  mkdir -p "$d"
  # Clear directory contents (including hidden entries; keep . and ..)
  rm -rf -- "$d"/* "$d"/.[!.]* "$d"/..?* 2>/dev/null || true
}

echo "src: $SRC_DIR"
echo "dst: $DST_DIR"
echo "dry-run: $DRY_RUN"

if [[ "$DRY_RUN" -eq 0 && "$CONFIRM" -ne 1 ]]; then
  echo "Nothing will be deleted by default. Add --yes to confirm, or --dry-run to preview."
  exit 1
fi

# Handle a few directory name mismatches (e.g. dst=calvc vs src=cavlc)
declare -A SRC_NAME_ALIASES=(
  [calvc]="cavlc"
)

for dst_x in "$DST_DIR"/*; do
  [[ -d "$dst_x" ]] || continue
  x="$(basename "$dst_x")"

  src_name="${SRC_NAME_ALIASES[$x]-$x}"
  src_x="$SRC_DIR/$src_name"

  if [[ ! -d "$src_x" ]]; then
    echo "Warning: source directory missing, skipping: $src_x (dst child $x)"
    continue
  fi

  dst_act_out="$dst_x/ACT_out"
  dst_merge_out_binary="$dst_x/merge_out_binary"
  dst_merge_binary_log="$dst_x/merge_binary.log"

  echo "Processing subdirectory: $x"

  run_cmd mkdir -p "$dst_act_out" "$dst_merge_out_binary"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] clear: $dst_act_out"
    echo "[dry-run] clear: $dst_merge_out_binary"
    echo "[dry-run] remove: $dst_merge_binary_log (if present)"
  else
    clear_dir_contents "$dst_act_out"
    clear_dir_contents "$dst_merge_out_binary"
    rm -f "$dst_merge_binary_log" 2>/dev/null || true
  fi

  # Copy all of src_x into dst_x/ACT_out (rsync preserves attributes; dst was cleared)
  if [[ "$DRY_RUN" -eq 1 ]]; then
    run_cmd rsync -a --delete "$src_x"/ "$dst_act_out"/
  else
    rsync -a --delete "$src_x"/ "$dst_act_out"/
  fi
done

echo "Done."

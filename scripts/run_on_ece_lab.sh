#!/usr/bin/env bash
set -e

# usage:
#   bash scripts/run_on_ece_lab.sh /path/to/EE567-Final-Project
# if no path is given, use current dir

ROOT_DIR="$1"
if [[ -z "$ROOT_DIR" ]]; then
  ROOT_DIR="$PWD"
fi

cd "$ROOT_DIR"

echo "[info] host=$(hostname)"
echo "[info] cwd=$PWD"

# ── Git LFS: the SWVF .gz files are stored via Git LFS ──
# ECE lab machines may not have git-lfs pre-installed, so install it into
# a local prefix if needed, then pull the real data files.
_ensure_git_lfs() {
  if command -v git-lfs >/dev/null 2>&1; then
    echo "[lfs] git-lfs already available"
    return 0
  fi

  # check if user-local copy exists (from a previous run)
  local LOCAL_BIN="$HOME/.local/bin"
  if [[ -x "$LOCAL_BIN/git-lfs" ]]; then
    export PATH="$LOCAL_BIN:$PATH"
    echo "[lfs] using $LOCAL_BIN/git-lfs"
    return 0
  fi

  echo "[lfs] git-lfs not found – installing to $LOCAL_BIN ..."
  mkdir -p "$LOCAL_BIN"

  local LFS_VER="3.6.1"
  local LFS_TAR="git-lfs-linux-amd64-v${LFS_VER}.tar.gz"
  local LFS_URL="https://github.com/git-lfs/git-lfs/releases/download/v${LFS_VER}/${LFS_TAR}"

  curl -fSL "$LFS_URL" -o "/tmp/$LFS_TAR"
  tar xzf "/tmp/$LFS_TAR" -C /tmp
  cp /tmp/git-lfs-${LFS_VER}/git-lfs "$LOCAL_BIN/git-lfs"
  chmod +x "$LOCAL_BIN/git-lfs"
  rm -rf "/tmp/$LFS_TAR" "/tmp/git-lfs-${LFS_VER}"

  export PATH="$LOCAL_BIN:$PATH"
  echo "[lfs] installed git-lfs $(git-lfs version)"
}

# Check whether the .gz files are LFS pointers (tiny text) instead of real data.
# A real gzip file starts with bytes 1f 8b; an LFS pointer starts with "version ".
_lfs_pointer_check() {
  local f="$1"
  if [[ ! -f "$f" ]]; then return 1; fi
  local magic
  magic=$(head -c 2 "$f" | xxd -p 2>/dev/null || echo "0000")
  if [[ "$magic" != "1f8b" ]]; then
    return 0   # looks like a pointer
  fi
  return 1     # real gzip
}

_pull_lfs_if_needed() {
  # only run if at least one .gz looks like an LFS pointer
  local need_pull=false
  for gz in SWVF_*.txt.gz; do
    if [[ -f "$gz" ]] && _lfs_pointer_check "$gz"; then
      need_pull=true
      break
    fi
  done

  if $need_pull; then
    echo "[lfs] SWVF .gz files are LFS pointers – pulling real data ..."
    _ensure_git_lfs
    git lfs install --local
    git lfs pull
    echo "[lfs] pull complete"
  else
    echo "[lfs] SWVF data files look OK (real gzip)"
  fi
}

_pull_lfs_if_needed

# ── Python setup ──
if ! command -v python3 >/dev/null 2>&1; then
  echo "[err] python3 not found"
  exit 1
fi

python3 -V

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# accept either .txt.gz or .txt data files
if [[ ! -f "SWVF_1_22.txt.gz" ]] && [[ ! -f "SWVF_1_22.txt" ]]; then
  echo "[err] missing SWVF files in repo root"
  echo "[err] place SWVF_1_22.txt(.gz) ... SWVF_67_88.txt(.gz) here"
  exit 1
fi

mkdir -p output
python run_pipeline.py

echo "[done] outputs are in $PWD/output"

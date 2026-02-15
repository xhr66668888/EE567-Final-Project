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

if [[ ! -f "SWVF_1_22.txt.gz" ]]; then
  echo "[err] missing SWVF files in repo root"
  exit 1
fi

mkdir -p output
python run_pipeline.py

echo "[done] outputs are in $PWD/output"

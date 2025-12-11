#!/usr/bin/env bash
set -euo pipefail

echo "== SAM2 setup script =="

# --------- Paths & config ---------
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$THIS_DIR"

PYTHON=${PYTHON:-python3}
VENV_DIR=${VENV_DIR:-.venv}
SAM2_REPO_DIR=${SAM2_REPO_DIR:-"$THIS_DIR/sam2_repo"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"$SAM2_REPO_DIR/checkpoints"}

# Official public checkpoint base URL (Meta)
BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/072824/"

# --------- 1. Create / activate venv ---------
if [ ! -d "$VENV_DIR" ]; then
  echo "➡ Creating virtualenv in $VENV_DIR"
  "$PYTHON" -m venv "$VENV_DIR"
else
  echo "✓ Virtualenv already exists: $VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "➡ Upgrading pip"
pip install --upgrade pip

# --------- 2. Install project requirements ---------
if [ -f "$THIS_DIR/requirements.txt" ]; then
  echo "➡ Installing project requirements"
  pip install -r "$THIS_DIR/requirements.txt"
else
  echo "⚠️  No requirements.txt found in $THIS_DIR, skipping"
fi

# --------- 3. Clone / update & install SAM2 repo ---------
if [ ! -d "$SAM2_REPO_DIR/.git" ]; then
  echo "➡ Cloning SAM2 repo into $SAM2_REPO_DIR"
  rm -rf "$SAM2_REPO_DIR" 2>/dev/null || true
  git clone https://github.com/facebookresearch/segment-anything-2.git "$SAM2_REPO_DIR"
else
  echo "✓ SAM2 repo already present: $SAM2_REPO_DIR"
  echo "➡ Updating SAM2 repo (git fetch/pull)"
  git -C "$SAM2_REPO_DIR" fetch --all --tags
  if [ -n "${SAM2_REV:-}" ]; then
    echo "➡ Checking out SAM2 revision: $SAM2_REV"
    git -C "$SAM2_REPO_DIR" checkout "$SAM2_REV"
  else
    git -C "$SAM2_REPO_DIR" pull --ff-only || echo "⚠️  SAM2 repo pull failed, continuing with existing checkout"
  fi
fi

# Always (re)install SAM2 requirements if present
if [ -f "$SAM2_REPO_DIR/requirements.txt" ]; then
  echo "➡ Installing SAM2 requirements"
  pip install -r "$SAM2_REPO_DIR/requirements.txt"
fi

echo "➡ Installing SAM2 in editable mode"
pip install -e "$SAM2_REPO_DIR"

# --------- 4. Download SAM2 checkpoints ---------
if [ -x "$SAM2_REPO_DIR/checkpoints/download_ckpts.sh" ]; then
  echo "➡ Running official checkpoint downloader"
  bash "$SAM2_REPO_DIR/checkpoints/download_ckpts.sh"
else
  echo "❌ checkpoints/download_ckpts.sh not found or not executable"
  exit 1
fi

echo
echo "✅ SAM2 setup complete!"
echo "  • Venv:          $VENV_DIR"
echo "  • SAM2 repo:     $SAM2_REPO_DIR"
echo "  • Checkpoints:   $CHECKPOINT_DIR"
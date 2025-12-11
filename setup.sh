#!/usr/bin/env bash
set -euo pipefail

echo "== SAM2 setup script =="

# --------- Paths & config ---------
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$THIS_DIR"

PYTHON=${PYTHON:-python3}
VENV_DIR=${VENV_DIR:-.venv}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"$THIS_DIR/checkpoints"}
SAM2_REPO_DIR=${SAM2_REPO_DIR:-"$THIS_DIR/segment-anything-2"}

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

# --------- 3. Clone & install SAM2 repo ---------
if [ ! -d "$SAM2_REPO_DIR" ]; then
  echo "➡ Cloning SAM2 repo into $SAM2_REPO_DIR"
  git clone https://github.com/facebookresearch/segment-anything-2.git "$SAM2_REPO_DIR"
else
  echo "✓ SAM2 repo already present: $SAM2_REPO_DIR"
fi

echo "➡ Installing SAM2 in editable mode"
pip install -e "$SAM2_REPO_DIR"

# --------- 4. Download SAM2 checkpoints ---------
mkdir -p "$CHECKPOINT_DIR"

download_ckpt() {
  local fname="$1"
  local url="${BASE_URL}${fname}"

  if [ -f "$CHECKPOINT_DIR/$fname" ]; then
    echo "✓ $fname already exists, skipping"
  else
    echo "⬇ Downloading $fname"
    wget -q --show-progress -O "$CHECKPOINT_DIR/$fname" "$url" \
      || { echo "❌ Failed to download $fname from $url"; exit 1; }
  fi
}

echo "➡ Downloading SAM2 checkpoints into $CHECKPOINT_DIR"
download_ckpt "sam2_hiera_tiny.pt"
download_ckpt "sam2_hiera_small.pt"
download_ckpt "sam2_hiera_base_plus.pt"
download_ckpt "sam2_hiera_large.pt"

echo
echo "✅ SAM2 setup complete!"
echo "  • Venv:          $VENV_DIR"
echo "  • SAM2 repo:     $SAM2_REPO_DIR"
echo "  • Checkpoints:   $CHECKPOINT_DIR"
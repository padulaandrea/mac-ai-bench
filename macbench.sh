#!/bin/sh
# mac-ai-bench: run a real local AI video studio flow on this Mac and time it.
#
#   curl -sL https://raw.githubusercontent.com/padulaandrea/mac-ai-bench/main/macbench.sh | sh
#   sh /path/to/mac-ai-bench/macbench.sh        (from a copied folder, e.g. AirDrop)
#
# Installs a temporary Python, the two exact tools (mlx-gen 0.38.0, ltx-2-mlx 0.18.0) and the two real models
# (FLUX.2-klein-4B, LTX-2.5 8-bit: about 57 GB of downloads) into ONE temporary folder, runs five real jobs
# (3 pictures, 2 videos), prints the timings, offers to upload them, and deletes the folder. Needs ~65 GB free.
#
# Options (environment): MACBENCH_KEEP=1 keep the folder; MACBENCH_HF_HOME=<dir> use models already downloaded there;
# MACBENCH_UPLOAD=0/1 skip the upload question; MACBENCH_SAVE=<dir> also save results.json there.
set -e
case "$(uname -m)" in arm64) ;; *) echo "This needs an Apple Silicon Mac (M1 or later)."; exit 1;; esac
LTX_COMMIT=14a3ddd6054736beb45b4d836135fa4d44490531   # ltx-2-mlx 0.18.0, as installed in the studio
D=$(mktemp -d "${TMPDIR:-/tmp}/macbench.XXXXXX")
if [ "${MACBENCH_KEEP:-0}" = 1 ]; then echo "Keeping $D"; else trap 'rm -rf "$D"' EXIT INT TERM HUP; fi
HF="${MACBENCH_HF_HOME:-$D/hf}"
if [ -z "$MACBENCH_HF_HOME" ]; then
  FREE=$(df -g "$D" | awk 'NR==2 {print $4}')
  if [ "$FREE" -lt 65 ]; then echo "Only $FREE GB free; this needs about 65 GB (models 57 GB + tools)."; exit 1; fi
fi
echo "Working in $D (deleted at the end)."
echo "1/4 Python and uv..."
# XDG_CONFIG_HOME: the installer's receipt goes into the temp folder too, not ~/.config/uv
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$D/bin" UV_NO_MODIFY_PATH=1 XDG_CONFIG_HOME="$D/config" sh >/dev/null 2>&1
export UV_CACHE_DIR="$D/cache" UV_PYTHON_INSTALL_DIR="$D/python"
UV="$D/bin/uv"
echo "2/4 Benchmark kit..."
HERE=$(cd "$(dirname "$0")" 2>/dev/null && pwd)
[ -z "$MACBENCH_KIT" ] && [ -f "$HERE/flow/jobs.json" ] && MACBENCH_KIT="$HERE"   # run from a copied folder
if [ -n "$MACBENCH_KIT" ]; then mkdir "$D/kit"; cp -R "$MACBENCH_KIT/flow" "$D/kit/flow"; else
  mkdir "$D/kit"; curl -LsSf https://codeload.github.com/padulaandrea/mac-ai-bench/tar.gz/main | tar xz -C "$D/kit" --strip-components 1
fi
echo "3/4 The studio's tools (exact versions)..."
mkdir "$D/ltx"; curl -LsSf "https://codeload.github.com/dgrauet/ltx-2-mlx/tar.gz/$LTX_COMMIT" | tar xz -C "$D/ltx" --strip-components 1
(cd "$D/ltx" && "$UV" sync --quiet --frozen --python 3.12)
"$UV" venv --quiet --python 3.12 "$D/pic"
"$UV" pip install --quiet --python "$D/pic/bin/python" -r "$D/kit/flow/requirements-picture.txt"
echo "4/4 Models and jobs..."
"$D/pic/bin/python" -u "$D/kit/flow/run.py" --kit "$D/kit/flow" --work "$D/work" --hf-home "$HF" \
  --mlxgen "$D/pic/bin/mlxgen" --ltx "$D/ltx/.venv/bin/ltx-2-mlx" "$@"
echo "Done. Temporary files are being deleted. (If a run was ever interrupted, clean-up.sh removes any leftovers.)"

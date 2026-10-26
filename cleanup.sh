#!/bin/sh
# mac-ai-bench clean-up: removes anything a benchmark run may have left behind (normally nothing: every run deletes
# its own temporary folder). Use it after a run was cut off, e.g. the Mac crashed or was force-quit.
#
#   curl -sL https://raw.githubusercontent.com/padulaandrea/mac-ai-bench/main/cleanup.sh | sh
found=0
for p in $(pgrep -f "macbench\.[A-Za-z0-9]{6}/" 2>/dev/null); do kill "$p" 2>/dev/null && found=1; done
[ "$found" = 1 ] && { echo "Stopped a running benchmark."; sleep 2; }
for d in "${TMPDIR:-/tmp}"/macbench.* /tmp/macbench.* /private/tmp/macbench.*; do
  [ -d "$d" ] || continue
  size=$(du -sh "$d" 2>/dev/null | cut -f1)
  rm -rf "$d" && echo "Deleted $d ($size)"; found=1
done
# early versions wrote uv's install receipt to ~/.config/uv: remove it only if it points into a benchmark folder
r="$HOME/.config/uv/uv-receipt.json"
if [ -f "$r" ] && grep -q "macbench\." "$r"; then
  rm -f "$r"; rmdir "$HOME/.config/uv" 2>/dev/null; rmdir "$HOME/.config" 2>/dev/null; echo "Deleted $r"; found=1
fi
[ "$found" = 1 ] && echo "Clean." || echo "Nothing to clean: no benchmark files on this Mac."

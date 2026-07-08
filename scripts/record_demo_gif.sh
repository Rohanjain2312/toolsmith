#!/usr/bin/env bash
# Record a screen region and produce an optimized GIF of the ToolSmith Space demo.
# Human-run only — Claude Code cannot drive a screen recording or browser interaction.
#
# Prerequisites:
#   - ffmpeg installed (macOS: `brew install ffmpeg`)
#   - The ToolSmith HF Space open in a browser window, positioned where you want it
#   - macOS: System Settings -> Privacy & Security -> Screen Recording -> allow your terminal
#
# What to record (per README.md's "Demo GIF placeholder" / plan §9):
#   1. Land on the Replay tab (default) — click through 1-2 curated SFT-vs-GRPO trajectory pairs
#   2. Switch to the Live tab — type a task, hit Run, let a step or two stream in
#   Keep it under ~20s total; GIFs are for a README hero, not a full walkthrough.
#
# Usage:
#   scripts/record_demo_gif.sh [duration_seconds] [output_path]
#   scripts/record_demo_gif.sh 20 docs/demo.gif
#
# macOS-specific (avfoundation). On Linux, swap the -f/-i capture args for x11grab
# (e.g. -f x11grab -i :0.0+100,200) and re-run.

set -euo pipefail

DURATION="${1:-20}"
OUTPUT_PATH="${2:-docs/demo.gif}"
RAW_VIDEO="$(mktemp -t toolsmith_demo_raw).mp4"
PALETTE="$(mktemp -t toolsmith_demo_palette).png"
FPS=12

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found. Install it first: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)." >&2
  exit 1
fi

echo "Listing available capture devices (find your screen's index below, e.g. '[2] Capture screen 0'):"
ffmpeg -f avfoundation -list_devices true -i "" 2>&1 | grep -A20 "AVFoundation video devices" || true

read -r -p "Screen device index to record (from the list above): " DEVICE_INDEX

echo "Recording ${DURATION}s from device ${DEVICE_INDEX} in 5 seconds — switch to your browser window now."
sleep 5

ffmpeg -y -f avfoundation -framerate "$FPS" -i "${DEVICE_INDEX}:none" -t "$DURATION" "$RAW_VIDEO"

echo "Generating an optimized color palette..."
ffmpeg -y -i "$RAW_VIDEO" -vf "fps=${FPS},scale=800:-1:flags=lanczos,palettegen" "$PALETTE"

echo "Encoding the final GIF..."
mkdir -p "$(dirname "$OUTPUT_PATH")"
ffmpeg -y -i "$RAW_VIDEO" -i "$PALETTE" \
  -filter_complex "fps=${FPS},scale=800:-1:flags=lanczos[x];[x][1:v]paletteuse" \
  "$OUTPUT_PATH"

rm -f "$RAW_VIDEO" "$PALETTE"
echo "Wrote ${OUTPUT_PATH} ($(du -h "$OUTPUT_PATH" | cut -f1))."
echo "Next: update README.md's demo GIF placeholder to '![ToolSmith demo](${OUTPUT_PATH})'."

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <target-root>"
  exit 1
fi

TARGET_ROOT="$1"
mkdir -p "$TARGET_ROOT"

if [[ ! -d "$TARGET_ROOT/F5-TTS" ]]; then
  git clone https://github.com/SWivid/F5-TTS.git "$TARGET_ROOT/F5-TTS"
fi

if [[ ! -d "$TARGET_ROOT/MegaTTS3" ]]; then
  git clone https://github.com/bytedance/MegaTTS3.git "$TARGET_ROOT/MegaTTS3"
fi

echo "Backends fetched under: $TARGET_ROOT"

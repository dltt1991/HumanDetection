#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
mkdir -p "$root/models"
destination="$root/models/picodet_s_320_person.onnx"
temporary=$(mktemp "$root/models/.picodet_s_320_person.onnx.XXXXXX")
trap 'rm -f "$temporary"' EXIT HUP INT TERM

curl -fL --retry 3 \
  https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_s_320_coco_lcnet.onnx \
  -o "$temporary"

size=$(wc -c < "$temporary" | tr -d '[:space:]')
[ "$size" = 4781217 ] || {
  echo "model size mismatch: expected 4781217 bytes, got $size" >&2
  exit 1
}

if command -v sha256sum >/dev/null 2>&1; then
  digest=$(sha256sum "$temporary" | awk '{print $1}')
else
  digest=$(shasum -a 256 "$temporary" | awk '{print $1}')
fi
[ "$digest" = f9c671308fe7c20e618c7e2b5f88cf7460443cfcb1ef4b628d00d95f0f79758e ] || {
  echo "model SHA-256 mismatch: got $digest" >&2
  exit 1
}

mv -f "$temporary" "$destination"
trap - EXIT HUP INT TERM
echo "verified model: $destination"

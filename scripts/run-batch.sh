#!/bin/sh

# Check arguments
if [ "$#" -lt 4 ]; then
    echo "Usage: $0 SCRIPT VIEWER INPUT_DIR OUTPUT_DIR"
    exit 1
fi

# Variables setup
WD="$(mktemp -d)"

SCRIPT="$(realpath "$1")"
shift

VIEWER="$(realpath "$1")"
shift

INPUT_DIR="$(realpath "$1")"
shift

OUTPUT_DIR="$(realpath "$1")"
shift

if [ ! -d "${OUTPUT_DIR}" ]; then
    mkdir -p "${OUTPUT_DIR}"
fi

# Running detection
for src_image in $INPUT_DIR/*.jpg; do
    python "$SCRIPT" "$src_image" --directory "$WD" > /dev/null
    base=$(basename "$src_image")
    csv="${WD}/${base%.*}.csv"
    python "$VIEWER" "$src_image" "$csv" "${OUTPUT_DIR}/${base}"
    echo "Wrote resulting image to ${OUTPUT_DIR}/${base}"
done

# Cleanup
rm -r "$WD"

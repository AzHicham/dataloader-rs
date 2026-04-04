#!/usr/bin/env bash
set -euo pipefail

echo "Editing files with given version: $1"

semantic-release-cargo prepare "$1"

#!/usr/bin/env sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

if ! python -m fibre_sight; then
    printf '%s\n' 'Fibre Sight could not start. Activate the fibre-sight environment, then run this file again.' >&2
    exit 1
fi

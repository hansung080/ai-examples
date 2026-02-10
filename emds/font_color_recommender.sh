#!/usr/bin/env bash

CURRENT_DIR="$(dirname "$(readlink -f "$0")")"
"$CURRENT_DIR/../.venv/bin/python3" "$CURRENT_DIR/background_brightness_classifier/cli.py"

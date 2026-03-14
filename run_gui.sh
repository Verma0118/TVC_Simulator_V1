#!/bin/bash
cd "$(dirname "$0")"
export QT_QPA_PLATFORM=cocoa
"./.venv/bin/python" tvc3d_gui_v2.py

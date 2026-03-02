#!/usr/bin/env bash
set -euo pipefail

D=/opt/dymola-2026x-x86_64
export DYMOLA="$D"
export LM_LICENSE_FILE=27000@kpfiler4.intra.dlr.de
export QT_QPA_PLATFORM=offscreen
export LD_LIBRARY_PATH="$D/bin/lib64:$D/bin/lib:$D/bin64:${LD_LIBRARY_PATH:-}"
export DYMOLA_USERHOME="$HOME/.dassaultsystemes/dymola/2026x"
export TMPDIR="$HOME/tmp"
mkdir -p "$DYMOLA_USERHOME" "$TMPDIR"

source .venv/bin/activate
python dymp_api_test.py

#!/usr/bin/env bash
set -euo pipefail
/opt/kg/bin/install_pring_runtime.sh
exec streamlit run /opt/kg/streamlit/app.py --server.address=0.0.0.0 --server.port=8501

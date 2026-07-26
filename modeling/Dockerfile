FROM python:3.11-slim

WORKDIR /opt/kg/modeling

ARG TORCH_VERSION=2.5.1
ARG TORCH_GEOMETRIC_VERSION=2.6.1
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG PYG_WHEEL_URL=https://data.pyg.org/whl/torch-2.5.1+cpu.html

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --upgrade pip setuptools wheel

COPY modeling/requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

# Install a pinned PyTorch/PyG stack. pyg_lib, torch_scatter and
# torch_sparse are compiled extensions and must come from the wheel index
# matching the installed torch. torch_sparse imports torch_scatter at runtime.
RUN python -m pip install --no-cache-dir \
      "torch==${TORCH_VERSION}" \
      --index-url "${PYTORCH_INDEX_URL}" \
    && python -m pip install --no-cache-dir \
      "torch-geometric==${TORCH_GEOMETRIC_VERSION}" \
    && python -m pip install --no-cache-dir --no-index \
      pyg_lib torch_scatter torch_sparse \
      -f "${PYG_WHEEL_URL}"

# Fail during image build rather than later at prediction time if the binary
# extensions do not match Python/PyTorch or LinkNeighborLoader is unavailable.
RUN python -m pip check \
    && python -c "import torch, torch_geometric, pyg_lib, torch_scatter, torch_sparse; from torch_geometric.data import HeteroData; from torch_geometric.loader import LinkNeighborLoader; d=HeteroData(); print('torch:', torch.__version__); print('torch cuda:', torch.version.cuda); print('torch_geometric:', torch_geometric.__version__); print('pyg_lib:', getattr(pyg_lib, '__version__', 'installed')); print('torch_scatter:', getattr(torch_scatter, '__version__', 'installed')); print('torch_sparse:', getattr(torch_sparse, '__version__', 'installed')); print('HeteroData:', type(d).__name__); print('LinkNeighborLoader:', LinkNeighborLoader.__name__)"

COPY scripts/install_pring_runtime.sh /opt/kg/bin/install_pring_runtime.sh
COPY modeling/pring_modeling ./pring_modeling
COPY modeling/stages ./stages
COPY modeling/notebooks ./notebooks
COPY modeling/scripts /opt/kg/modeling/scripts
COPY modeling/scripts/run_modeling.sh /opt/kg/bin/run_modeling.sh
RUN chmod +x /opt/kg/bin/*.sh

ENV PYTHONPATH=/opt/kg/modeling \
    PYTHONUNBUFFERED=1

# Repeat the PyG preflight after copying application code. This detects runtime
# import-order or module-shadowing problems that an earlier dependency-only
# check cannot see. It also verifies the actual FastAPI import path.
RUN python -c "from pring_modeling.pyg_runtime import initialize_pyg_runtime; s=initialize_pyg_runtime(); assert s.ready, f'{s.error_type}: {s.error}'; print(s.versions)" \
    && python -c "import pring_modeling.prediction_api; print('prediction API import: ok')"

CMD ["/opt/kg/bin/run_modeling.sh"]

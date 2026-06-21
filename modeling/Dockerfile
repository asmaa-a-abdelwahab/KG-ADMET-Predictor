FROM python:3.11-slim

WORKDIR /opt/kg/modeling

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --upgrade pip

COPY modeling/requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY scripts/install_pring_runtime.sh /opt/kg/bin/install_pring_runtime.sh
COPY modeling/pring_modeling ./pring_modeling
COPY modeling/stages ./stages
COPY modeling/notebooks ./notebooks
COPY modeling/scripts/run_modeling.sh /opt/kg/bin/run_modeling.sh
RUN chmod +x /opt/kg/bin/*.sh

ENV PYTHONPATH=/opt/kg/modeling
CMD ["/opt/kg/bin/run_modeling.sh"]

FROM python:3.11-slim-bookworm

WORKDIR /opt/kg/streamlit

# wkhtmltopdf is intentionally not installed here because it is a large,
# distro-sensitive system dependency. The app now handles missing PDF export
# gracefully and still provides the HTML report.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --upgrade pip

COPY streamlit/requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY scripts/install_pring_runtime.sh /opt/kg/bin/install_pring_runtime.sh
COPY streamlit/app.py ./app.py
COPY streamlit/utils ./utils
COPY streamlit/images ./images
COPY streamlit/scripts/start_streamlit.sh /opt/kg/bin/start_streamlit.sh
RUN chmod +x /opt/kg/bin/*.sh

EXPOSE 8501
ENV PYTHONPATH=/opt/kg/streamlit
CMD ["/opt/kg/bin/start_streamlit.sh"]

FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/kopredict
COPY pyproject.toml README.md requirements.txt ./
COPY src ./src
RUN pip install --no-cache-dir .

ENV KOPREDICT_DATA_DIR=/data
VOLUME ["/data"]
ENTRYPOINT ["kopredict"]

FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/kopredict
COPY pyproject.toml README.md requirements.txt ./
COPY src ./src
COPY metadata ./metadata
RUN pip install --no-cache-dir .

# DepMap files are mounted at ./data (this working directory).
ENV KOPREDICT_DATA_DIR=data
VOLUME ["/opt/kopredict/data"]
ENTRYPOINT ["kopredict"]

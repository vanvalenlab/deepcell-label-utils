FROM ubuntu:20.04

WORKDIR /deepcell-label-utils

WORKDIR /data-registry

RUN apt-get update && apt-get install -y \
	git python3-pip libbz2-dev liblzma-dev && \
	rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["/bin/bash"]
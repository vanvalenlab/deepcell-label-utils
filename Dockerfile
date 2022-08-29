FROM ubuntu:20.04

WORKDIR /deepcell-label-utils

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
	git python3-pip python3-opencv libbz2-dev liblzma-dev && \
	rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["/bin/bash"]

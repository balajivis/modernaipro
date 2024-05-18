# Use the official Miniconda image as the base image
FROM continuumio/miniconda3

RUN apt-get update && apt-get install build-essential -y

COPY ./mitraaiBase.yml /tmp/
RUN conda env create -f /tmp/mitraaiBase.yml
COPY ./mitraaiGradio.yml /tmp/
RUN conda env create -f /tmp/mitraaiGradio.yml

WORKDIR /workspace

RUN apt-get update && apt-get install git -y && apt-get install curl -y

RUN curl -fsSL https://ollama.com/install.sh | sh

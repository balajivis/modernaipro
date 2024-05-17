# Use the official Miniconda image as the base image
FROM continuumio/miniconda3

COPY ./mitraaiAlpha.yml /tmp/
COPY ./mitraaiGradio.yml /tmp/

RUN apt-get update && apt-get install build-essential -y
RUN conda env update --name base --file /tmp/mitraaiAlpha.yml
RUN conda env create -f /tmp/mitraaiGradio.yml

WORKDIR /workspace

RUN apt-get update && apt-get install git -y && apt-get install curl -y

RUN curl -fsSL https://ollama.com/install.sh | sh

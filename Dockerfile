FROM ubuntu:bionic

WORKDIR /root

COPY . .

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev
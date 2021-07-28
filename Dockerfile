FROM ubuntu:20.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN apt-get update

RUN apt install -y gcc

#set noninteractive installation
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Zurich
#install tzdata package
RUN apt-get install -y tzdata

RUN apt install -y libgl1-mesa-glx

ADD requirements.txt .
ADD setup.sh .

RUN ./setup.sh

RUN apt-get install -y libfontconfig
RUN apt-get install -y libxrender-dev
RUN apt-get install -y libxkbcommon-x11-0
RUN apt-get install -y libdbus-1-3
RUN apt-get install -y x11-xserver-utils 

#RUN useradd -r -u 11659 -ms /bin/bash seashelluser
RUN useradd seashelluser

#ENV HOME /home/seashelluser

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#    && mkdir /home/seashelluser/.conda \
#    && chown seashelluser /home/seashelluser/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

#USER seashelluser
#RUN conda --version
#RUN conda install python==3.7.10

#RUN conda init
#RUN pip3 install -r requirements.txt

WORKDIR /home/seashelluser

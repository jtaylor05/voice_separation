FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

ADD requirements.txt .
ADD src /src

RUN apt-get update && apt-get install -y git
RUN apt-get install -y openssh-client
RUN apt-get install sudo
RUN apt-get install wget
RUN pip3 -q install pip --upgrade
RUN pip install --upgrade pip

RUN pip install -r requirements.txt
RUN pip install dvc
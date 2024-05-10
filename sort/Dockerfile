FROM python:3.9

RUN pip install --upgrade pip
RUN pip install --upgrade pip setuptools
RUN python --version
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN mkdir tracker_demo
RUN cd tracker_demo

COPY requirements.txt /tracker_demo

WORKDIR /tracker_demo

RUN pip3 install -r /tracker_demo/requirements.txt
RUN pip3 install scikit-image
RUN pip3 install lap


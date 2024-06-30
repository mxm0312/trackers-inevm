FROM python:3.9

RUN pip install --upgrade pip
RUN pip install --upgrade pip setuptools
RUN python --version
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN mkdir byte_tracker_demo
RUN cd byte_tracker_demo

COPY . /byte_tracker_demo

WORKDIR /byte_tracker_demo

RUN pip3 install numpy
RUN pip3 install -r /byte_tracker_demo/requirements.txt

CMD ["python", "main.py"]
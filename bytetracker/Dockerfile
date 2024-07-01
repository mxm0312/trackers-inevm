FROM python:3.9

RUN pip install --upgrade pip
RUN pip install --upgrade pip setuptools
RUN python --version
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /

COPY .. /

CMD cd /bytetracker

RUN pip3 install numpy
RUN pip3 install -r ./requirements.txt

ENTRYPOINT ["python","./main.py"]
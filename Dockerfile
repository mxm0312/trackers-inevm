FROM python:3.9

#RUN pip install --upgrade pip
#RUN pip install --upgrade pip setuptools
RUN python --version
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y && apt-get install -y docker.io

WORKDIR /

COPY common/ ./common/
COPY bytetracker/ ./bytetracker/
COPY embeddings/ ./embeddings/

RUN pip3 install numpy==1.25
RUN pip3 install --default-timeout=100 -r /bytetracker/requirements.txt

ENTRYPOINT ["python","/bytetracker/main.py"]

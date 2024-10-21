FROM python:3.11-slim

RUN apt-get update && apt-get -y install build-essential

COPY requirements.txt /opt/app/requirements.txt
RUN pip3 install -r /opt/app/requirements.txt

COPY src /opt/app/src
COPY main.py /opt/app/main.py

WORKDIR /opt/app
RUN mkdir log

CMD python3 main.py

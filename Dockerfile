FROM python:3.9.18-slim

WORKDIR /root

COPY requirements.txt /root

RUN pip install --no-cache-dir -r requirements.txt

COPY . /root

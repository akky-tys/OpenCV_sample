FROM python:3.6

RUN apt-get update -y && apt-get install -y libopencv-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install numpy tensorflow opencv-python==3.4.4.19
RUN apt-get update && apt-get -y install vim
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools



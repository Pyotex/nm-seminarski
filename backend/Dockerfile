FROM python

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1  

WORKDIR /usr/local/app/

RUN apt-get update && apt-get install -y \
    python3-opencv \
    libopencv-dev

COPY . .
RUN pip install --upgrade pip && pip install -r requirements.txt

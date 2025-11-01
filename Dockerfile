FROM python:3.11.9-slim-bullseye

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

WORKDIR /app/src

ENV IS_DOCKER_CONTAINER=1 \
    VALIDATION_ARGS_FILE=validation_config_cpu.json.example

CMD ["bash", "-lc", "./start.sh --mode ${START_MODE:-dual}"]

FROM python:3.11.9-slim-bullseye

WORKDIR /app

COPY . .

RUN pip3 install -r requirements.txt

EXPOSE 8000

ENV IS_DOCKER_CONTAINER=1

CMD ["python", "-m", "src.entrypoint"]

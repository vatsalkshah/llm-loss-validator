FROM python:3.11.9-slim-bullseye

WORKDIR /app

COPY . .

RUN pip3 install -r requirements.txt

WORKDIR /app/src

# Environment defaults for CPU-based workers
ENV IS_DOCKER_CONTAINER=1
ENV WORKER_MODE=validation
ENV MAX_CACHE_SIZE_GB=50
ENV TELEMETRY_ENABLED=false

# Expose port for inference API (when running in inference or dual mode)
EXPOSE 8000

CMD ["sh", "-c", "bash start.sh --hf_token ${HF_TOKEN} --flock_api_key ${FLOCK_API_KEY} --task_id ${TASK_ID} --validation_args_file validation_config_cpu.json.example"]

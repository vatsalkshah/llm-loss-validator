import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AppConfig:
    port: int
    heartbeat_interval_sec: int
    node_location: Optional[str]
    model_cache_max: int
    hf_token: Optional[str]
    flock_api_key: Optional[str]
    validation_task_id: Optional[str]


def get_config() -> AppConfig:
    return AppConfig(
        port=int(os.getenv("PORT", "8000")),
        heartbeat_interval_sec=int(os.getenv("HEARTBEAT_INTERVAL_SEC", "10")),
        node_location=os.getenv("NODE_LOCATION"),
        model_cache_max=int(os.getenv("MODEL_CACHE_MAX", "2")),
        hf_token=os.getenv("HF_TOKEN"),
        flock_api_key=os.getenv("FLOCK_API_KEY"),
        validation_task_id=os.getenv("VALIDATION_TASK_ID"),
    )

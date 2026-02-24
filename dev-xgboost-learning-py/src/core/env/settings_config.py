
import os
import yaml
from functools import lru_cache
from core.env.environment_config import AppConfig
from pathlib import Path
import re


def expand_env_variables(content: str) -> str:
    # Reemplaza ${VAR} por el valor real de os.environ["VAR"]
    return re.sub(r'\$\{([^}]+)}', lambda m: os.getenv(m.group(1), ""), content)


@lru_cache(maxsize=1)
def load_config() -> AppConfig:
    app_env = os.getenv("APP_ENV")
    with open(Path(f"resources/env/env_{app_env}.yaml"), "r") as f:
        content = f.read()
    expand_content = expand_env_variables(content)
    data = yaml.safe_load(expand_content)
    return AppConfig(**data)

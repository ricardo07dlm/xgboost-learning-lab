from pydantic import BaseModel


class AppConfigEnvironment(BaseModel):
    url_vector_db: str


class AppConfig(BaseModel):
    application: AppConfigEnvironment

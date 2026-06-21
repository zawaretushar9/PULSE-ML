from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "AutoML Platform"
    API_V1_STR: str = "/api"
    
    # CORS settings
    ALLOWED_ORIGINS: list = ["http://localhost:5173", "https://automl-platform-frontend.vercel.app"]
    
    # MLflow settings
    MLFLOW_TRACKING_URI: str = "sqlite:///mlflow.db"
    MLFLOW_ARTIFACT_ROOT: str = "./mlruns"
    
    # File upload settings
    UPLOAD_DIR: str = "uploads"
    
    class Config:
        case_sensitive = True

settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

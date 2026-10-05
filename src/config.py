import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    
    MODEL_DIR: str = os.getenv("MODEL_DIR", "models")
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    SENTIMENT_MODEL: str = os.getenv("SENTIMENT_MODEL", "distilbert").lower()
    
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @property
    def model_path(self) -> Path:
        return self.PROJECT_ROOT / self.MODEL_DIR
        
    @property
    def data_path(self) -> Path:
        return self.PROJECT_ROOT / self.DATA_DIR

settings = Settings()

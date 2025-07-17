import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    MODEL_PATH: str = os.path.join(os.path.dirname(__file__), "..", "model")
    LOOKBACK_PERIOD: int = 30
    FORECAST_DAYS: int = 90

    # Hour bins for peak hour analysis
    HOUR_BINS = ["7-9", "9-11", "11-13", "13-15", "15-17"]
    HOUR_BINS_LABELS = ["7-9 AM", "9-11 AM", "11-1 PM", "1-3 PM", "3-5 PM"]


settings = Settings()

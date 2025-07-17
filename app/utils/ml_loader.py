import os
import joblib
import logging
from typing import Optional, Tuple, Any

logger = logging.getLogger(__name__)


class MLAssetsLoader:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.feature_scaler = None
        self.total_visits_scaler = None
        self._load_assets()

    def _load_assets(self):
        """Load all ML assets with proper error handling"""
        try:
            # Load model
            model_file = os.path.join(self.model_path, "vms_trend_prediction.keras")
            if os.path.exists(model_file):
                from keras.models import load_model

                self.model = load_model(model_file)
                logger.info(f"✅ Model loaded from: {model_file}")
            else:
                raise FileNotFoundError(f"Model file not found: {model_file}")

            # Load scalers
            feature_scaler_file = os.path.join(self.model_path, "feature_scaler.gz")
            total_visits_scaler_file = os.path.join(
                self.model_path, "total_visits_scaler.gz"
            )

            if os.path.exists(feature_scaler_file):
                self.feature_scaler = joblib.load(feature_scaler_file)
                logger.info("✅ Feature scaler loaded")
            else:
                raise FileNotFoundError(
                    f"Feature scaler not found: {feature_scaler_file}"
                )

            if os.path.exists(total_visits_scaler_file):
                self.total_visits_scaler = joblib.load(total_visits_scaler_file)
                logger.info("✅ Total visits scaler loaded")
            else:
                raise FileNotFoundError(
                    f"Total visits scaler not found: {total_visits_scaler_file}"
                )

            logger.info("✅ All ML assets loaded successfully")

        except Exception as e:
            logger.error(f"❌ Error loading ML assets: {e}")
            raise

    def is_ready(self) -> bool:
        """Check if all assets are loaded"""
        return all(
            [
                self.model is not None,
                self.feature_scaler is not None,
                self.total_visits_scaler is not None,
            ]
        )

    def get_assets(self) -> Tuple[Any, Any, Any]:
        """Get all loaded assets"""
        return self.model, self.feature_scaler, self.total_visits_scaler

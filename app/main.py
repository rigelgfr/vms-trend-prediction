from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager

from app.config import settings
from model.schemas import ForecastResponse, HealthResponse, DataTestResponse
from app.services.database import DatabaseService
from app.services.prediction import PredictionService
from app.utils.ml_loader import MLAssetsLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables
ml_loader = None
db_service = None
prediction_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global ml_loader, db_service, prediction_service

    logger.info("Starting up VMS Trend Prediction API...")

    # Check database URL
    if not settings.DATABASE_URL:
        logger.error("DATABASE_URL not found in environment variables")
        raise RuntimeError("DATABASE_URL not found in .env file")

    # Initialize ML assets
    try:
        ml_loader = MLAssetsLoader(settings.MODEL_PATH)
        if not ml_loader.is_ready():
            raise RuntimeError("ML assets could not be loaded")

        # Initialize services
        db_service = DatabaseService(settings.DATABASE_URL)
        model, feature_scaler, total_visits_scaler = ml_loader.get_assets()
        prediction_service = PredictionService(
            model, feature_scaler, total_visits_scaler
        )

        logger.info("✅ All systems ready - Database and ML assets loaded successfully")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down VMS Trend Prediction API...")


# Create FastAPI app
app = FastAPI(
    title="VMS Trend Prediction API",
    description="An API to forecast visitor trends for the next 90 days.",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "VMS Trend Prediction API is running", "status": "healthy"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if ml_loader and ml_loader.is_ready() else "initializing",
        model_loaded=ml_loader is not None and ml_loader.model is not None,
        scalers_loaded=ml_loader is not None
        and ml_loader.feature_scaler is not None
        and ml_loader.total_visits_scaler is not None,
        database_configured=bool(settings.DATABASE_URL),
    )


@app.get("/predict", response_model=ForecastResponse)
async def get_90_day_prediction():
    """
    Fetches historical visitor data and returns 90-day forecast
    """
    if not ml_loader or not ml_loader.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Service is still initializing. Please try again in a moment.",
        )

    try:
        logger.info("Starting prediction process...")

        # Step 1: Fetch and aggregate data
        logger.info(f"Fetching last {settings.LOOKBACK_PERIOD} days of data...")
        aggregated_df = db_service.fetch_and_aggregate_data(
            num_days=settings.LOOKBACK_PERIOD
        )

        if aggregated_df.empty:
            raise HTTPException(
                status_code=404, detail="No historical data found in the database"
            )

        if len(aggregated_df) < settings.LOOKBACK_PERIOD:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data: Need {settings.LOOKBACK_PERIOD} days, got {len(aggregated_df)} days",
            )

        logger.info(f"✅ Data fetched successfully: {len(aggregated_df)} days")

        # Step 2: Prepare sequence for model
        logger.info("Preparing feature sequence...")
        initial_sequence, feature_names, last_real_date, holidays_obj = (
            prediction_service.prepare_sequence_from_df(aggregated_df)
        )
        logger.info("✅ Feature sequence prepared")

        # Step 3: Generate forecast
        logger.info("Generating 90-day forecast...")
        forecast = prediction_service.recursive_forecast(
            initial_sequence=initial_sequence,
            feature_names=feature_names,
            last_real_date=last_real_date,
            indonesia_holidays=holidays_obj,
            num_days=settings.FORECAST_DAYS,
        )
        logger.info("✅ Forecast generated successfully")

        return ForecastResponse(
            status="success",
            forecast=forecast,
            metadata={
                "total_predictions": len(forecast),
                "historical_data_points": len(aggregated_df),
                "last_historical_date": last_real_date.strftime("%Y-%m-%d"),
                "forecast_start_date": forecast[0]["Date"] if forecast else None,
                "forecast_end_date": forecast[-1]["Date"] if forecast else None,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Internal server error during prediction: {str(e)}"
        )


@app.get("/test-data", response_model=DataTestResponse)
async def test_data_fetch():
    """
    Test endpoint to check if data can be fetched from database
    """
    if not db_service:
        raise HTTPException(status_code=503, detail="Database service not initialized")

    try:
        # Fetch just 5 days for testing
        test_df = db_service.fetch_and_aggregate_data(num_days=5)

        return DataTestResponse(
            status="success",
            data_shape=test_df.shape,
            columns=list(test_df.columns),
            sample_data=test_df.head(3).to_dict() if not test_df.empty else None,
            date_range={
                "start": str(test_df.index.min()) if not test_df.empty else None,
                "end": str(test_df.index.max()) if not test_df.empty else None,
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error testing data fetch: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

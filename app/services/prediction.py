import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays
from typing import Tuple, List, Dict, Any
import logging
from app.config import settings

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self, model, feature_scaler, total_visits_scaler):
        self.model = model
        self.feature_scaler = feature_scaler
        self.total_visits_scaler = total_visits_scaler

    def prepare_sequence_from_df(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, list, pd.Timestamp, holidays.HolidayBase]:
        """
        Engineers features from aggregated DataFrame for LSTM model
        """
        logger.info("Engineering features for model...")

        # Get feature names from scaler
        feature_names = self.feature_scaler.get_feature_names_out()

        # Initialize features DataFrame
        features = pd.DataFrame(index=df.index, columns=feature_names).fillna(0)

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Initialize holidays
        indonesia_holidays = holidays.Indonesia(
            years=range(df.index.year.min(), df.index.year.max() + 2)
        )

        # Feature engineering for each date
        for i, date in enumerate(df.index):
            try:
                # Lag features
                lag_1_date = date - timedelta(days=1)
                lag_7_date = date - timedelta(days=7)

                # Handle lag features with bounds checking
                if lag_1_date in df.index:
                    features.loc[date, "Lag_1"] = df.loc[lag_1_date, "Total_Visits"]
                else:
                    features.loc[date, "Lag_1"] = 0

                if lag_7_date in df.index:
                    features.loc[date, "Lag_7"] = df.loc[lag_7_date, "Total_Visits"]
                else:
                    features.loc[date, "Lag_7"] = 0

                # Rolling average (7-day window ending the day before)
                if i >= 1:
                    window_start = max(0, i - 7)
                    window_data = df.iloc[window_start:i]["Total_Visits"]
                    features.loc[date, "Rolling_Avg_7"] = (
                        window_data.mean() if len(window_data) > 0 else 0
                    )
                else:
                    features.loc[date, "Rolling_Avg_7"] = 0

                # Categorical features
                features.loc[date, f"Day_of_Week_{date.weekday()}"] = 1
                features.loc[date, f"Month_{date.month}"] = 1

                # Payday period (25-31 and 1-3 of month)
                features.loc[date, "Is_Payday_Period"] = (
                    1 if date.day in list(range(25, 32)) + list(range(1, 4)) else 0
                )

                # Holiday features
                features.loc[date, "Is_Holiday"] = (
                    1 if date.date() in indonesia_holidays else 0
                )

                # Days until next holiday
                future_holidays = [
                    hd for hd in indonesia_holidays.keys() if hd >= date.date()
                ]
                if future_holidays:
                    next_holiday = min(future_holidays)
                    features.loc[date, "Days_until_Holiday"] = (
                        next_holiday - date.date()
                    ).days
                else:
                    features.loc[date, "Days_until_Holiday"] = 365

            except Exception as e:
                logger.error(f"Error processing date {date}: {e}")
                continue

        # Scale features
        try:
            scaled_features = self.feature_scaler.transform(features)
            logger.info(f"✅ Features engineered: {scaled_features.shape}")
        except Exception as e:
            logger.error(f"Error scaling features: {e}")
            raise

        last_real_date = df.index.max()

        return scaled_features, list(feature_names), last_real_date, indonesia_holidays

    def recursive_forecast(
        self,
        initial_sequence: np.ndarray,
        feature_names: list,
        last_real_date: pd.Timestamp,
        indonesia_holidays: holidays.HolidayBase,
        num_days: int = 90,
    ) -> List[Dict[str, Any]]:
        """
        Performs recursive forecasting for specified number of days
        """
        logger.info(f"Starting {num_days}-day recursive forecast...")

        # Initialize tracking variables
        current_sequence = initial_sequence.copy().tolist()

        # Convert initial sequence back to unscaled visits for lag calculation
        try:
            unscaled_history = list(
                self.total_visits_scaler.inverse_transform(
                    initial_sequence[:, 0].reshape(-1, 1)
                ).flatten()
            )
        except Exception as e:
            logger.error(f"Error converting initial sequence: {e}")
            raise

        predictions = []
        current_date = last_real_date

        for day in range(num_days):
            try:
                # Prepare input for model (last LOOKBACK_PERIOD days)
                input_sequence = np.array(
                    current_sequence[-settings.LOOKBACK_PERIOD :]
                ).reshape(1, settings.LOOKBACK_PERIOD, -1)

                # Make prediction
                pred_scaled_visits, pred_probs_hours = self.model.predict(
                    input_sequence, verbose=0
                )

                # Convert back to original scale
                pred_visits = self.total_visits_scaler.inverse_transform(
                    pred_scaled_visits
                )[0][0]

                # Apply business logic for weekends/holidays
                next_date = current_date + timedelta(days=1)
                is_weekend = next_date.weekday() >= 5
                is_holiday = next_date.date() in indonesia_holidays

                if is_weekend or is_holiday:
                    pred_visits = max(0, np.random.randint(0, 3))
                else:
                    pred_visits = max(0, pred_visits)

                # Get top 2 peak hours
                top_2_indices = np.argsort(pred_probs_hours[0])[-2:]
                top_2_labels = [
                    settings.HOUR_BINS_LABELS[idx] for idx in sorted(top_2_indices)
                ]

                # Store prediction
                predictions.append(
                    {
                        "Date": next_date.strftime("%Y-%m-%d"),
                        "Visits": int(round(pred_visits)),
                        "Peak_1": top_2_labels[0],
                        "Peak_2": (
                            top_2_labels[1]
                            if len(top_2_labels) > 1
                            else top_2_labels[0]
                        ),
                    }
                )

                # Update tracking variables
                current_date = next_date
                unscaled_history.append(pred_visits)

                # Engineer features for next prediction
                lag_1 = unscaled_history[-2] if len(unscaled_history) >= 2 else 0
                lag_7 = unscaled_history[-8] if len(unscaled_history) >= 8 else 0

                rolling_window = (
                    unscaled_history[-8:-1]
                    if len(unscaled_history) >= 8
                    else unscaled_history[:-1]
                )
                rolling_avg_7 = (
                    np.mean(rolling_window) if len(rolling_window) > 0 else 0
                )

                # Create new feature vector
                new_feature_vector = pd.Series(index=feature_names, dtype=float).fillna(
                    0
                )
                new_feature_vector["Lag_1"] = lag_1
                new_feature_vector["Lag_7"] = lag_7
                new_feature_vector["Rolling_Avg_7"] = rolling_avg_7
                new_feature_vector[f"Day_of_Week_{current_date.weekday()}"] = 1
                new_feature_vector[f"Month_{current_date.month}"] = 1
                new_feature_vector["Is_Payday_Period"] = (
                    1
                    if current_date.day in list(range(25, 32)) + list(range(1, 4))
                    else 0
                )
                new_feature_vector["Is_Holiday"] = (
                    1 if current_date.date() in indonesia_holidays else 0
                )

                # Days until next holiday
                future_holidays = [
                    hd for hd in indonesia_holidays.keys() if hd >= current_date.date()
                ]
                if future_holidays:
                    next_holiday = min(future_holidays)
                    new_feature_vector["Days_until_Holiday"] = (
                        next_holiday - current_date.date()
                    ).days
                else:
                    new_feature_vector["Days_until_Holiday"] = 365

                # Scale and add to sequence
                scaled_new_vector = self.feature_scaler.transform(
                    new_feature_vector.to_frame().T
                )
                current_sequence.append(scaled_new_vector[0])

            except Exception as e:
                logger.error(f"Error in forecast day {day + 1}: {e}")
                # Add fallback prediction
                predictions.append(
                    {
                        "Date": (current_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                        "Visits": 0,
                        "Peak_1": "N/A",
                        "Peak_2": "N/A",
                    }
                )
                current_date += timedelta(days=1)

        logger.info(f"✅ Forecast complete: {len(predictions)} predictions generated")
        return predictions

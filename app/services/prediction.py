import pandas as pd
import numpy as np
from datetime import timedelta
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
    ) -> Tuple[np.ndarray, list, pd.Timestamp, holidays.HolidayBase, pd.Series]:
        """
        Engineers features from aggregated DataFrame for LSTM model.
        This logic now EXACTLY mirrors the training script and handles all categories.
        """
        logger.info("Engineering features for model using vectorized operations...")

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        features = pd.DataFrame(index=df.index)
        features["Lag_1"] = df["Total_Visits"].shift(1)
        features["Lag_7"] = df["Total_Visits"].shift(7)
        features["Rolling_Avg_7"] = df["Total_Visits"].shift(1).rolling(window=7).mean()


        day_of_week_categorical = pd.Categorical(
            df.index.dayofweek, categories=range(7)
        )
        day_dummies = pd.get_dummies(day_of_week_categorical, prefix="Day_of_Week")
        day_dummies.index = df.index

        month_categorical = pd.Categorical(df.index.month, categories=range(1, 13))
        month_dummies = pd.get_dummies(month_categorical, prefix="Month")
        month_dummies.index = df.index

        features = features.join(day_dummies)
        features = features.join(month_dummies)

        features["Is_Payday_Period"] = df.index.day.isin(
            list(range(25, 32)) + list(range(1, 4))
        ).astype(int)

        indonesia_holidays = holidays.Indonesia(
            years=range(df.index.year.min(), df.index.year.max() + 2)
        )
        features["Is_Holiday"] = df.index.isin(indonesia_holidays).astype(int)

        holiday_dates = sorted(indonesia_holidays.keys())
        next_holiday_dates = []
        for d in df.index.date:
            future_holidays = [hd for hd in holiday_dates if hd >= d]
            if future_holidays:
                next_holiday_dates.append(min(future_holidays))
            else:
                next_holiday_dates.append(d + timedelta(days=365))

        features["Days_until_Holiday"] = (
            pd.to_datetime(next_holiday_dates) - df.index
        ).days

        features_original_index = features.index
        features = features.dropna()
        logger.info(
            f"Dropped {len(features_original_index) - len(features)} rows with NaN values."
        )

        historical_visits = df.loc[features.index, "Total_Visits"]

        feature_names_from_scaler = self.feature_scaler.get_feature_names_out()
        features = features[feature_names_from_scaler]

        try:
            scaled_features = self.feature_scaler.transform(features)
            logger.info(f"✅ Features engineered and scaled: {scaled_features.shape}")
        except Exception as e:
            logger.error(
                f"Error scaling features. Check column order and names. Error: {e}"
            )
            raise

        initial_sequence = scaled_features[-settings.LOOKBACK_PERIOD :]
        historical_visits_for_loop = historical_visits[-settings.LOOKBACK_PERIOD :]
        last_real_date = historical_visits_for_loop.index.max()

        return (
            initial_sequence,
            list(feature_names_from_scaler),
            last_real_date,
            indonesia_holidays,
            historical_visits_for_loop,
        )

    def recursive_forecast(
        self,
        initial_sequence: np.ndarray,
        historical_visits: pd.Series,
        feature_names: list,
        last_real_date: pd.Timestamp,
        indonesia_holidays: holidays.HolidayBase,
        num_days: int = 90,
    ) -> List[Dict[str, Any]]:
        logger.info(f"Starting {num_days}-day recursive forecast...")

        current_sequence = initial_sequence.copy().tolist()
        unscaled_history = historical_visits.tolist()

        predictions = []
        current_date = last_real_date

        for day in range(num_days):
            try:
                input_sequence = np.array(
                    current_sequence[-settings.LOOKBACK_PERIOD :]
                ).reshape(1, settings.LOOKBACK_PERIOD, -1)

                pred_scaled_visits, pred_probs_hours = self.model.predict(
                    input_sequence, verbose=0
                )
                pred_visits = self.total_visits_scaler.inverse_transform(
                    pred_scaled_visits
                )[0][0]

                next_date = current_date + timedelta(days=1)
                pred_visits = max(0, pred_visits)

                top_2_indices = np.argsort(pred_probs_hours[0])[-2:]
                top_2_labels = [
                    settings.HOUR_BINS_LABELS[idx] for idx in sorted(top_2_indices)
                ]

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

                current_date = next_date
                unscaled_history.append(pred_visits)

                lag_1 = unscaled_history[-2] if len(unscaled_history) >= 2 else 0
                lag_7 = unscaled_history[-8] if len(unscaled_history) >= 8 else 0

                rolling_window = (
                    unscaled_history[-8:-1]
                    if len(unscaled_history) >= 8
                    else unscaled_history[:-1]
                )
                rolling_avg_7 = np.mean(rolling_window) if rolling_window else 0

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

                future_holidays = [
                    hd for hd in indonesia_holidays.keys() if hd >= current_date.date()
                ]
                if future_holidays:
                    next_holiday = min(future_holidays)
                    new_vector_days_until_holiday = (
                        next_holiday - current_date.date()
                    ).days
                else:
                    new_vector_days_until_holiday = 365
                new_feature_vector["Days_until_Holiday"] = new_vector_days_until_holiday

                scaled_new_vector = self.feature_scaler.transform(
                    new_feature_vector.to_frame().T[feature_names]
                )
                current_sequence.append(scaled_new_vector[0])

            except Exception as e:
                logger.error(f"Error in forecast day {day + 1}: {e}", exc_info=True)
                predictions.append(
                    {
                        "Date": (current_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                        "Visits": 0,
                        "Peak_1": "N/A",
                        "Peak_2": "N/A",
                    }
                )
                current_date += timedelta(days=1)
                unscaled_history.append(0)
                current_sequence.append(np.zeros_like(current_sequence[-1]))

        logger.info(f"✅ Forecast complete: {len(predictions)} predictions generated")
        return predictions

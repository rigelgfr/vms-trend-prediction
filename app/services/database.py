import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta, timezone
import logging
from app.config import settings

logger = logging.getLogger(__name__)


class DatabaseService:
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.engine = create_engine(db_url)

    def fetch_and_aggregate_data(self, num_days: int) -> pd.DataFrame:
        """
        Connects to database, fetches visit records, and aggregates into daily summary
        """
        logger.info(f"Fetching data for last {num_days} days...")

        try:
            # Calculate date range
            end_date = datetime.now(timezone.utc).date()
            start_date = end_date - timedelta(days=num_days - 1)

            logger.info(f"Date range: {start_date} to {end_date}")

            # Query with proper parameterization
            query = text(
                """
                SELECT entry_start_date, check_in_time
                FROM visit
                WHERE entry_start_date >= :start_date 
                AND entry_start_date <= :end_date
                ORDER BY entry_start_date
            """
            )

            with self.engine.connect() as connection:
                df = pd.read_sql(
                    query,
                    connection,
                    params={"start_date": start_date, "end_date": end_date},
                )

            logger.info(f"Raw data fetched: {len(df)} records")

            if df.empty:
                logger.warning("No data found in database for the given date range")
                return pd.DataFrame()

            # Data cleaning and transformation
            df["entry_start_date"] = pd.to_datetime(df["entry_start_date"])

            # Handle timezone conversion more safely
            if df["check_in_time"].dtype == "object":
                df["check_in_time"] = pd.to_datetime(df["check_in_time"])

            # Convert to Jakarta timezone if not already timezone-aware
            if df["check_in_time"].dt.tz is None:
                df["check_in_time"] = df["check_in_time"].dt.tz_localize("UTC")

            df["check_in_time"] = df["check_in_time"].dt.tz_convert("Asia/Jakarta")

            # Step 1: Daily visit counts
            daily_visits = (
                df.groupby(df["entry_start_date"].dt.date)
                .size()
                .reset_index(name="Total_Visits")
            )
            daily_visits = daily_visits.rename(columns={"entry_start_date": "Date"})
            daily_visits = daily_visits.set_index("Date")

            # Step 2: Peak hour analysis
            df["hour_bin"] = pd.cut(
                df["check_in_time"].dt.hour,
                bins=[7, 9, 11, 13, 15, 17],
                labels=settings.HOUR_BINS,
                right=False,
                include_lowest=True,
            )

            # Count visits by date and hour bin
            peak_hours_pivot = (
                df.groupby([df["entry_start_date"].dt.date, "hour_bin"])
                .size()
                .unstack(fill_value=0)
            )

            # Find top 2 peak hours for each day
            def get_top_hours(row):
                top_2 = row.nlargest(2)
                return pd.Series(
                    [
                        top_2.index[0] if len(top_2) > 0 else "N/A",
                        top_2.index[1] if len(top_2) > 1 else "N/A",
                    ],
                    index=["Top_Hour_1", "Top_Hour_2"],
                )

            top_hours_df = peak_hours_pivot.apply(get_top_hours, axis=1)

            # Step 3: Create complete date range and combine data
            full_date_range = pd.date_range(
                start=start_date, end=end_date, freq="D"
            ).date

            # Initialize result DataFrame
            aggregated_df = pd.DataFrame(index=full_date_range)
            aggregated_df.index.name = "Date"

            # Join all data
            aggregated_df = aggregated_df.join(daily_visits, how="left")
            aggregated_df = aggregated_df.join(top_hours_df, how="left")

            # Fill missing values
            aggregated_df["Total_Visits"] = (
                aggregated_df["Total_Visits"].fillna(0).astype(int)
            )
            aggregated_df["Top_Hour_1"] = aggregated_df["Top_Hour_1"].fillna("N/A")
            aggregated_df["Top_Hour_2"] = aggregated_df["Top_Hour_2"].fillna("N/A")

            logger.info(f"Data aggregation complete: {len(aggregated_df)} days")
            logger.info(
                f"Total visits range: {aggregated_df['Total_Visits'].min()} - {aggregated_df['Total_Visits'].max()}"
            )

            return aggregated_df

        except Exception as e:
            logger.error(f"Error in fetch_and_aggregate_data: {str(e)}")
            raise e

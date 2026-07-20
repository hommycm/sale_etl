from pyspark.sql import SparkSession
import logging
import sys
import traceback
from datetime import datetime
import os

from src.etl_pipeline import *

def setup_logging():
    """Basic logging setup"""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/etl_run_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def main():

    os.makedirs('logs', exist_ok=True)
    os.makedirs('data/processed/orders', exist_ok=True)

    logger = setup_logging()
    logger.info("Starting Sales ETL Pipeline")

    # Track runtime
    start_time = datetime.now()

    try:
        # Initialize Spark
        spark = create_spark_session()
        logger.info("Spark session created")

        # Extract
        raw_df = extract_sales_data(spark, "data/raw/ecommerce_sales_analytics_5000.csv")
        logger.info(f"Extracted {raw_df.count()} raw records")

        # Transform
        clean_df = transform_orders(raw_df)
        logger.info(f"Transformed to {clean_df.count()} clean records")

        # Load
        output_path = "data/processed/orders"
        load_to_csv(spark, clean_df, output_path)


        # Create summary
        summary = create_summary_report(clean_df)

        # Calculate runtime
        runtime = (datetime.now() - start_time).total_seconds()
        logger.info(f"Pipeline completed successfully in {runtime:.2f} seconds")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise

    finally:
        spark.stop()
        logger.info("Spark session closed")

if __name__ == "__main__":
    main()
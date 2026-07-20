import logging
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *

logger = logging.getLogger(__name__)

def create_spark_session() : 
    return SparkSession.builder \
    .appName("Sales ETL") \
    .master("local[*]") \
    .getOrCreate()

def extract_sales_data(spark, input_path) :
    
    logger.info(f"Reading sales data from {input_path}")


    df = spark.read.csv(
        input_path,
        header=True,
        inferSchema=True,
        mode="PERMISSIVE"
    )
    
    total_records = df.count()
    logger.info(f"Found {total_records} total records")

    return df

def standardize_dates(df):

    fmt1 = to_date(try_to_timestamp(col("order_date"), lit("yyyy-MM-dd")))
    fmt2 = to_date(try_to_timestamp(col("order_date"), lit("MM/dd/yyyy")))
    fmt3 = to_date(try_to_timestamp(col("order_date"), lit("M/d/yyyy")))
    fmt4 = to_date(try_to_timestamp(col("order_date"), lit("dd-MM-yyyy")))

    df = df.withColumn(
        "order_date_parsed",
        coalesce(fmt1, fmt2, fmt3, fmt4)
    )

    unparsed = df.filter(col("order_date_parsed").isNull()).count()

    if unparsed > 0:
        logger.warning(f"Could not parse {unparsed} dates")

    return df.drop("order_date")

def handle_duplicates(df):

    df_deduped = df.dropDuplicates(["order_id"])

    duplicate_count = df.count() - df_deduped.count()
    if duplicate_count > 0:
        logger.info(f"Removed {duplicate_count} duplicate orders")

    return df_deduped

def transform_orders(df):
    """Apply all transformations in sequence"""

    logger.info("Starting data transformation...")

    df = standardize_dates(df)
    df = handle_duplicates(df)

    df = df.withColumn(
        "quantity",
        when(col("quantity").isNotNull(), col("quantity").cast(IntegerType()))
        .otherwise(1)
    )

    df = df.withColumn("processing_date", current_date())

    df = df.withColumnRenamed("order_date_parsed", "order_date") \
           .withColumnRenamed("price_decimal", "unit_price")

    logger.info(f"Transformation complete. Final record count: {df.count()}")

    return df

def load_to_csv(spark, df, output_path):
    logger.info(f"Writing {df.count()} records to {output_path}")

    pandas_df = df.toPandas()

    import os
    os.makedirs(output_path, exist_ok=True)

    output_file = f"{output_path}/orders.csv"
    pandas_df.to_csv(output_file, index=False)

    logger.info(f"Successfully wrote {len(pandas_df)} records")
    logger.info(f"Output location: {output_file}")

    return len(pandas_df)

def create_summary_report(df):

    summary = {
        "total_orders": df.count(),
        "unique_customers": df.select("customer_id").distinct().count(),
        "unique_products": df.select("product_category").distinct().count(),
        "payment_methods": df.select("payment_method").distinct().count(),
        "total_revenue": df.agg(sum("revenue")).collect()[0][0],
        "date_range": (
            f"{df.agg(min('order_date')).collect()[0][0]} "
            f"to "
            f"{df.agg(max('order_date')).collect()[0][0]}"
        ),
    }

    region_summary = (
        df.groupBy("region")
          .agg(count("*").alias("number_of_orders"))
          .orderBy("region")
    )

    rating_summary = (
        df.groupBy("customer_rating")
          .agg(count("*").alias("number_of_orders"))
          .orderBy("customer_rating")
    )

    logger.info("\n========== ETL Summary Report ==========")

    for key, value in summary.items():
        logger.info(f"{key}: {value}")

    logger.info("\nOrders by Region")
    for row in region_summary.collect():
        logger.info(
            f"{row['region']}: {row['number_of_orders']} orders"
        )

    logger.info("\nOrders by Customer Rating")
    for row in rating_summary.collect():
        logger.info(
            f"Rating {row['customer_rating']}: {row['number_of_orders']} orders"
        )

    logger.info("========================================\n")

    return {
        "summary": summary,
        "region_summary": region_summary,
        "rating_summary": rating_summary,
    }

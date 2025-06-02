################################################################################################
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date
import pandas as pd
from pyspark.sql.functions import year
from pyspark.sql.window import Window
from pyspark.sql.functions import last, col
from pyspark.sql.functions import log, lag, col, year
from pyspark.sql.functions import mean, stddev
################################################################################################
spark = SparkSession.builder.appName("US1").getOrCreate()
################################################################################################

input1 = snakemake.input[0]
df_stocks = spark.read.csv(input1, header = True, inferSchema=True)
################################################################################################

input2 = snakemake.input[1]
df_annual = pd.read_csv(input2)
################################################################################################

reqd_cusips = df_annual["cusip"].unique().tolist()
df_filtered = df_stocks.select("datadate", "cusip", "prccd", "ajexdi", "trfd")
filtered_df = df_filtered.filter(df_filtered['cusip'].isin(reqd_cusips))
filtered_df = filtered_df.withColumn("datadate", to_date("datadate"))
filtered_df = filtered_df.withColumn("year", year("datadate"))
################################################################################################

w = Window.partitionBy("cusip", "year").orderBy("datadate").rowsBetween(Window.unboundedPreceding, 0)

cols_to_fill = ['prccd', 'ajexdi', 'trfd']

for col_name in cols_to_fill:
    filtered_df = filtered_df.withColumn(
        f"{col_name}_ffill",
        last(col(col_name), ignorenulls=True).over(w)
    )

filtered_df = filtered_df.withColumn(
    "daily_return_price",
    (col("prccd_ffill") / col("ajexdi_ffill")) * col("trfd_ffill")
)
################################################################################################

w = Window.partitionBy("cusip", "year").orderBy("datadate")

filtered_df = filtered_df.withColumn(
    "prev_price",
    lag("daily_return_price").over(w)
)

filtered_df = filtered_df.withColumn(
    "log_return",
    log(col("daily_return_price") / col("prev_price"))
)

filtered_df = filtered_df.filter(col("prev_price").isNotNull())
df_filtered = filtered_df.select("year", "cusip", "log_return")
################################################################################################

df_stats = df_filtered.groupBy("cusip", "year").agg(
    mean("log_return").alias("expected_return"),
    stddev("log_return").alias("volatility")
)

df_stats.toPandas().to_csv(snakemake.output[0])  # "Data/prepared_datasets/stocks_data_prepared.csv")
################################################################################################

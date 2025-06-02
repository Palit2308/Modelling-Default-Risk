from pyspark.sql import SparkSession
from Scripts.functions.create_labelled_data.load_labelled_data import load_labelled_data
from pyspark.sql import Window
from pyspark.sql.functions import col, lag, countDistinct, count
from pyspark.sql.functions import col, lit, sum as spark_sum, round as spark_round
spark = SparkSession.builder.appName("US1").getOrCreate()

input1 = snakemake.input[0]
input2 = snakemake.input[1]


df_annual = load_labelled_data(spark = spark, input_fs = input1, input_bk = input2)

# Step 1: Sort and define the window by cusip and year
window_spec = Window.partitionBy("cusip").orderBy("fyear")

# Step 2: Add a column for the default status in the previous year
df_with_lag = df_annual.withColumn("prev_default_status", lag("default_status").over(window_spec))

# Step 3: Filter for default events where previous year was null
default_events = df_with_lag.filter(
    (col("default_status") == "D") & (col("prev_default_status").isNull())
)

# Step 4: Count unique cusips with default events per year
default_counts = default_events.groupBy("fyear").agg(
    countDistinct("cusip").alias("No. of Default Events")
)

# Step 5: Count total observations per year
observation_counts = df_annual.groupBy("fyear").agg(
    count("*").alias("No. of Observations")
)

# Step 6: Join both summaries on fyear
summary = observation_counts.join(default_counts, on="fyear", how="left").fillna(0)

# Rename column to match desired output
summary = summary.withColumnRenamed("fyear", "Year")

summary_final = summary.orderBy("Year")

summary_final = summary_final.toPandas()

summary_final.to_csv(snakemake.output[0], index = False)

spark.stop()

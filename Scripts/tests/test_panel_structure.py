from pyspark.sql import SparkSession
from Scripts.functions.create_labelled_data.load_labelled_data import load_labelled_data
from pyspark.sql.functions import col, count, countDistinct

spark = SparkSession.builder.appName("US1").getOrCreate()

input1 = "Data/financials/financials_annual.gz"
input2 = "Data/bankruptcy_filings/bankruptcy_filings_cleaned.csv"


df_annual = load_labelled_data(spark, input1, input2)

check = df_annual.groupBy("fyear").agg(
    count("*").alias("total_rows"),
    countDistinct("cusip").alias("unique_cusips")
)


mismatch = check.filter(col("total_rows") != col("unique_cusips")).count()

assert mismatch == 0
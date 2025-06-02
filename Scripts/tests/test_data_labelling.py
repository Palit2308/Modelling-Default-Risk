#######################################################################################################
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import to_date, col, year, countDistinct, sum, when, count, lit, round, lag
import pandas as pd
from pyspark.sql.functions import col, lower,row_number
from pyspark.sql.functions import year, count, col, expr, greatest
from pyspark.sql import functions as F
from pyspark.sql.functions import col, to_date, concat_ws
from Scripts.functions.create_labelled_data.load_labelled_data import load_labelled_data
from Scripts.functions.create_labelled_data.create_df_matches import create_df_matching
from pyspark.sql.functions import collect_list, max
#######################################################################################################
spark = SparkSession.builder.appName("US1").getOrCreate()

input1 = "Data/financials/financials_annual.gz"
input2 = "Data/bankruptcy_filings/bankruptcy_filings_cleaned.csv"

df_cr = spark.read.csv(input2, header = True, inferSchema = True)

df_annual = load_labelled_data(spark, input1, input2)
df_matches = create_df_matching(spark, df_annual, df_cr)
#######################################################################################################
df_matches_filtered = df_matches.select("cusip", "YearFiled", "YearEmerged", "YearRefiled")

sample1 = df_matches_filtered.filter(
    col("YearFiled").isNotNull() & col("YearEmerged").isNull()
).limit(2)

sample2 = df_matches_filtered.filter(
    col("YearFiled").isNotNull() & col("YearEmerged").isNotNull() & col("YearRefiled").isNull()
).limit(2)

sample3 = df_matches_filtered.filter(
    col("YearFiled").isNotNull() & col("YearEmerged").isNotNull() & col("YearRefiled").isNotNull()
).limit(2)

sample_cusips = sample1.union(sample2).union(sample3)
#######################################################################################################
df_check = df_annual.join(sample_cusips, on="cusip", how="inner")
#######################################################################################################
result = df_check.filter(col("default_status") == "D") \
    .groupBy("cusip", "YearFiled", "YearEmerged", "YearRefiled") \
    .agg(max("fyear").alias("end_year"), collect_list("fyear").alias("default_status_years")
    ) \
    .orderBy("cusip")
#######################################################################################################

from pyspark.sql.functions import collect_list, max, col, udf
from pyspark.sql.types import BooleanType
# Step 2: Define the validation UDF
def years_within_end(years_list, end):
    return all(y <= end for y in years_list)

check_udf = udf(years_within_end, BooleanType())

# Step 3: Apply the check
validated = result.withColumn("valid", check_udf(col("default_status_years"), col("end_year")))

# Step 4: Count violations and assert
violations = validated.filter(col("valid") == False)

# Step 5: Raise error if any violations are found
assert violations.count() == 0, "Assertion failed: Some cusips have default_status_years beyond end_year."


spark.stop()


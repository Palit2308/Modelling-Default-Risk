# This script inspects whether the raw data is a well structured panel.
# Is there one observation per cusip-year?

# For years, we have 2 columns: datadate and fyear. datadate is the day that the financials were released, and 
# fyear is the financial year of accounting.

# To check whether the data is a good panel, the first step is to check if there is ONLY 1 cusip-fyear combination present.

# Since there are more than 1 cusip-fyear combinations, it is necessary to inspect, if there are more than one datadates
# in one year. If so, we must identify which is the dominant datadate for each fiscal year.

# Uniformly, throughout all years, most companies report their financials for a fiscal year YYYY, on the last date of the calender
# year YYYY-12-31. The output dominant_datadate_per_year.csv is structured as follows:

#                         fyear | datadate1 | datadate2 | ....... | dominant_datadate
#                         ------------------------------------------------------------

# This shows, in per year which dates have most observations. That date is the dominant datadate for that year. 

# The output table companies_per_year_dominant_date.csv is structured as follows:

#                              fyear | dominant_datadate | unique_cusips
#                              ------------------------------------------

# This table shows how many companies have reported their financial statements on the dominant datadates per year.

# Finally, if there are duplicate entries on some dominant datadates for some fyear-cusip, only the observation with the latest
# reporting date "apdedate" is kept. 

# This gives  a clean panel with unique fyear-cusip combinations. 

#######################################################################################################
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import to_date, col, year, countDistinct, sum, when, count, lit, round, lag
import pandas as pd
from pyspark.sql.functions import col, lower,row_number
from pyspark.sql.functions import year, count, col, expr, greatest
from pyspark.sql import functions as F
from pyspark.sql.functions import col, to_date, concat_ws
#######################################################################################################

input = snakemake.input[0]

# Start Spark session
spark = SparkSession.builder.appName("US1").getOrCreate()

df_annual = spark.read.csv(input, header=True, inferSchema=True)

#######################################################################################################

# IS THE DATAFRAME A PROPER PANEL? IS THERE 1 OBSERVATION PER CUSIP AND FYEAR?

duplicates = df_annual.groupBy("cusip", "fyear") \
    .count() \
    .filter(col("count") > 1) \
    .orderBy("cusip", "fyear")

print("number of duplicates in total are :" , duplicates.count())

#######################################################################################################

# WHAT IS THE DOMINANT DATE OF FINANCIAL STATEMENTS FOR EACH FISCAL YEAR?

grouped = df_annual.groupBy("fyear", "datadate").agg(count("*").alias("n_obs"))

pivoted = grouped.groupBy("fyear").pivot("datadate").agg(F.first("n_obs")).fillna(0)

datadate_cols = pivoted.columns[1:]

exprs = [F.struct(col(c).alias("count"), F.lit(c).alias("datadate")) for c in datadate_cols]
pivoted = pivoted.withColumn("dominant_struct", F.greatest(*exprs)) \
                 .withColumn("dominant_datadate", col("dominant_struct.datadate")) \
                 .drop("dominant_struct")

pivoted = pivoted.orderBy("fyear")


#######################################################################################################

# HOW MANY COMPANIES HAVE DATA ONLY ON THE DOMINANT DATADATES

df_annual = df_annual.withColumn(
    "dominant_datadate", to_date(concat_ws("-", col("fyear"), lit("12"), lit("31")))
)

df_filtered = df_annual.filter(col("datadate") == col("dominant_datadate"))

result = df_filtered.groupBy("fyear", "dominant_datadate") \
    .agg(countDistinct("cusip").alias("unique_cusips")) \
    .orderBy("fyear")


#######################################################################################################

# KEEPING ONLY THE DATA FOR DOMINANT DATADATES AND DROPPING DUPLICATES - KEEPING ONLY THE LATEST ACTUAL DATE
# BECOMES A FUNCTION IN THE FUNCTIONS FOLDER

df_annual = df_annual.withColumn(
    "dominant_datadate", to_date(concat_ws("-", col("fyear"), lit("12"), lit("31")))
)

df_conforming = df_annual.filter(col("datadate") == col("dominant_datadate"))

df_conforming = df_conforming.drop("dominant_datadate")

window_spec = Window.partitionBy("cusip", "fyear", "datadate").orderBy(col("apdedate").desc())

df_latest = df_conforming.withColumn("row_num", row_number().over(window_spec)) \
                         .filter(col("row_num") == 1) \
                         .drop("row_num")

#######################################################################################################

# NOW INSPECTING THE REVISED PANEL STRUCTURE - 1 YEAR CUSIP COMBINATION EACH

duplicates = df_latest.groupBy("cusip", "fyear") \
    .count() \
    .filter(col("count") > 1) \
    .orderBy("cusip", "fyear")

print("number of duplicates in total are :" , duplicates.count())

#######################################################################################################


pivoted_df = pivoted.toPandas()
pivoted_df.to_csv(snakemake.output[0], index = False)

result_df = result.toPandas()
result_df.to_csv(snakemake.output[1], index = False)

spark.stop()
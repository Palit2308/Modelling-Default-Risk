
# This script identifies the defaulting companies, identified by their CUSIP ids
# It also identifies the year of their default. 
# The output "defaulting_companies_year" is a table which looks like this:

#                      NameCorp | YearFiled | Chapter | cusip
#                      --------------------------------------

# Chapter is the identifier for either Chapter 7 or Chapter 11 Bankruptcy
    
# The output consists of ONLY the cusips which have been identified from the Financial Statements
# Thus it gives number of companies in OUR dataset that have filed for bankruptcy.

# The second output "yearly_cases_defaults" is a table which looks like this:

#                               Year | n_defaults
#                               ------------------

# Since the first ouput has only 1 Unique Cusip-Year combination, the yearly sum of the number of
# new defaults in each year.

#######################################################################################################
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import to_date, col, year, countDistinct, sum, when, count, lit, round, lag
import pandas as pd
from pyspark.sql.functions import col, lower,row_number
from Scripts.functions.create_labelled_data.create_clean_panel import clean_annual_panel
from Scripts.functions.create_labelled_data.remove_gaps import remove_gaps
#######################################################################################################

input1 = snakemake.input[0] 
input2 = snakemake.input[1] 

spark = SparkSession.builder.appName("US1").getOrCreate()

df_cr = spark.read.option("header", True).csv(input1)                         ## BANKRUPTCY FILINGS DATA
df_annual = spark.read.csv(input2, header=True, inferSchema=True)             ## FINANCIAL STATEMENTS YEARLY
df_annual = clean_annual_panel(df_annual)
df_annual = remove_gaps(df_annual)
#######################################################################################################

df_cr = df_cr.filter((col("Chapter") == "7") | (col("Chapter") == "11"))

df_cr= df_cr.filter(col("DateFiled").isNotNull())
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
df_cr = df_cr.withColumn("DateFiled", to_date("DateFiled", "M/d/yyyy"))
df_cr = df_cr.withColumn("DateEmerging", to_date("DateEmerging", "M/d/yyyy"))
df_cr = df_cr.withColumn("DateRefile", to_date("DateRefile", "M/d/yyyy"))

df_cr = df_cr.withColumn("YearFiled", year(col("DateFiled")))
df_cr = df_cr.withColumn("YearEmerged", year(col("DateEmerging")))
df_cr = df_cr.withColumn("YearRefiled", year(col("DateRefile")))

df_cr = df_cr.select("NameCorp", "Chapter", "DateFiled", "cusip", "YearFiled", "YearEmerged", "YearRefiled")

#######################################################################################################

df_annual = df_annual["conm", "tic", "cusip"]
df_unique_conms = df_annual.select("cusip").distinct()

#######################################################################################################

df_matches = df_cr.alias("cr").join(
    df_unique_conms.alias("an"),
    (lower(col("cr.cusip")).contains(lower(col("an.cusip")))) |
    (lower(col("an.cusip")).contains(lower(col("cr.cusip"))))
)

#######################################################################################################

window_spec = Window.partitionBy("an.cusip").orderBy(col("YearFiled").desc())

# Assign row numbers so the latest record gets row_number = 1
df_matches = df_matches.withColumn("row_num", row_number().over(window_spec))

# Filter to keep only the latest record per cusip
df_matches = df_matches.filter(col("row_num") == 1).drop("row_num")

df_matches = df_matches.select(
    col("an.cusip").alias("cusip"),
    col("cr.NameCorp"),
    col("cr.YearFiled"),
    col("cr.Chapter"),
    col("cr.YearEmerged"),
    col("cr.YearRefiled")
)

#######################################################################################################


df_defaults = df_matches.groupBy("YearFiled") \
    .agg(countDistinct("cusip").alias("n_defaults")) \
    .orderBy("YearFiled")


#######################################################################################################

df_matches = df_matches.toPandas()
df_matches.to_csv(snakemake.output[0], index = False)

df_defaults = df_defaults.toPandas()
df_defaults.to_csv(snakemake.output[1], index = False)

spark.stop()




#######################################################################################################
from pyspark.sql.window import Window
from pyspark.sql.functions import to_date, col, year, countDistinct, sum, when, count, lit, round, lag
import pandas as pd
from pyspark.sql.functions import col, lower,row_number
from pyspark.sql.functions import year, count, col, expr, greatest
from pyspark.sql import functions as F
from pyspark.sql.functions import col, to_date, concat_ws
#######################################################################################################

# FUNCTION FOR KEEPING ONLY THE DATA FOR DOMINANT DATADATES AND DROPPING DUPLICATES - KEEPING ONLY THE LATEST ACTUAL DATE

def clean_annual_panel(df_annual):

    df_annual = df_annual.withColumn(
        "dominant_datadate", to_date(concat_ws("-", col("fyear"), lit("12"), lit("31")))
    )

    df_conforming = df_annual.filter(col("datadate") == col("dominant_datadate"))

    df_conforming = df_conforming.drop("dominant_datadate")

    window_spec = Window.partitionBy("cusip", "fyear", "datadate").orderBy(col("apdedate").desc())

    df_latest = df_conforming.withColumn("row_num", row_number().over(window_spec)) \
                            .filter(col("row_num") == 1) \
                            .drop("row_num")
    
    return df_latest

######################################################################################################################
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, min, max, collect_set, array_sort, size, expr, array_distinct, when, sort_array
from pyspark.sql.types import ArrayType, IntegerType
from pyspark.sql.functions import when, size, lit
from pyspark.sql.functions import col, min, max, to_date,year, lower, row_number
from pyspark.sql.functions import col, expr
######################################################################################################################

def drop_selected_out_of_range_cusips(df_annual, df_matches):
    #######################################################################################################
    # USE THE DF_MATCHES TO IDENTIFY THE OUT OF RANGE CUSIPS
    #######################################################################################################
    df_defaults = df_matches.withColumnRenamed("YearFiled", "default_year")
    #######################################################################################################
    joined = df_annual.join(df_defaults.select("cusip", "default_year"), on="cusip", how="inner")
    #######################################################################################################
    summary = joined.groupBy("cusip", "default_year").agg(
        min("fyear").alias("start_year"),
        max("fyear").alias("end_year")
    )
    summary = summary.withColumn(
        "in_range",
        (col("default_year") >= col("start_year")) & (col("default_year") <= col("end_year"))
    )
    #######################################################################################################
    out_of_range_df = summary.filter(col("in_range") == False)
    out_of_range_df = out_of_range_df.withColumn(
        "distance", col("default_year") - col("end_year")
    )
    #######################################################################################################
    # DROPPING OUT OF RANGE CUSIPS WHICH WHOSE DISTANCE (DEFAULT YEAR - END YEAR) < 0 AND > 2.
    # IF DISTANCE IS < 0 FOR AN OUT OF RANGE CUSIP, THIS MEANS, THE DATA FOR THAT CUSIP STARTS AFTER THE 
    # COMPANY HAS DEFAULTED
    # IF DISTANCE > 2, THIS MEANS THAT THE DATA FOR THE COMPANY IS UNAVAILABLE SINCE MORE THAN 2 YEARS PRIOR
    # TO DEFAULT
    #######################################################################################################
    cusips_to_drop = out_of_range_df \
        .filter((col("distance") < 0) | (col("distance") > 2)) \
        .select("cusip") \
        .distinct()
    joined_filtered = df_annual.join(cusips_to_drop, on="cusip", how="left_anti")
        
    return joined_filtered
